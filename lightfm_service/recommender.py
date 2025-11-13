from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime
from pymongo import MongoClient
from collections import defaultdict

try:
	from lightfm import LightFM
	from scipy.sparse import csr_matrix
except Exception:  # allow runtime without LightFM in dev
	LightFM = None
	csr_matrix = None


@dataclass
class UserProfile:
	user_id: str
	experience: str
	skill_level: int
	age: int
	time_spent_coding: float
	motivation: str
	activity_level: str
	completion_rate: float
	engagement_score: float

	def to_dict(self) -> Dict:
		return {
			'userId': self.user_id,
			'experience': self.experience,
			'skillLevel': self.skill_level,
			'age': self.age,
			'timeSpentCoding': self.time_spent_coding,
			'motivation': self.motivation,
			'activityLevel': self.activity_level,
			'completionRate': self.completion_rate,
			'engagementScore': self.engagement_score,
		}


class LightFMRecommender:
	def __init__(self, mongo_uri: str = '', mongo_db: str = '', use_bootstrap_fallback: bool = True):
		self.mongo_uri = mongo_uri
		self.mongo_db = mongo_db
		self.use_bootstrap_fallback = use_bootstrap_fallback
		
		self._mongo_client = None
		self._db = None
		if mongo_uri and mongo_db:
			try:
				self._mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
				self._db = self._mongo_client[mongo_db]
				# Test connection
				self._mongo_client.server_info()
				print(f"[LightFM] Connected to MongoDB: {mongo_db}")
			except Exception as e:
				print(f"[LightFM] MongoDB connection failed: {e}, using bootstrap fallback")
				self._mongo_client = None
				self._db = None

		self._items = []
		self._item_index = {}
		self._user_index = {}
		self._user_id_map = {}  # Maps user_id string to index
		self._model: Optional[LightFM] = None
		self._last_trained = None
		self._min_interactions_for_training = 10  # Minimum interactions needed to train

		# Load data and train
		self._load_data_and_train()

	def _load_data_and_train(self) -> None:
		"""Load interactions and tasks from MongoDB, then train model"""
		try:
			if self._db is None:
				print("[LightFM] No MongoDB connection, using bootstrap")
				self._items = self._bootstrap_items()
				self._train_bootstrap_model()
				return

			# Load tasks from database
			tasks_collection = self._db.get_collection('tasks')
			tasks = list(tasks_collection.find(
				{'status': {'$ne': 'cancelled'}},  # Exclude cancelled tasks
				{'title': 1, 'difficulty': 1, 'category': 1, 'skills': 1, 'estimatedTime': 1}
			).limit(1000))  # Limit to prevent memory issues

			if not tasks or len(tasks) == 0:
				print("[LightFM] No tasks found in DB, using bootstrap")
				self._items = self._bootstrap_items()
				self._train_bootstrap_model()
				return

			# Build item catalog from real tasks
			self._items = []
			self._item_index = {}
			for task in tasks:
				item_id = str(task['_id'])
				item = {
					'id': item_id,
					'title': task.get('title', 'Untitled Task'),
					'difficulty': task.get('difficulty', 'medium'),
					'category': task.get('category', 'general'),
					'skills': task.get('skills', []),
					'minutes': task.get('estimatedTime', 30)
				}
				self._items.append(item)
				self._item_index[item_id] = len(self._items) - 1

			print(f"[LightFM] Loaded {len(self._items)} tasks from database")

			# Load user-task interactions
			interactions_collection = self._db.get_collection('usertaskinteractions')
			interactions = list(interactions_collection.find({
				'interactionWeight': {'$gt': 0}  # Only positive interactions
			}).limit(10000))  # Limit for performance

			if len(interactions) < self._min_interactions_for_training:
				print(f"[LightFM] Only {len(interactions)} interactions found (need {self._min_interactions_for_training}), using bootstrap")
				self._train_bootstrap_model()
				return

			# Build user index
			unique_user_ids = set(inter['userId'] for inter in interactions)
			self._user_id_map = {uid: idx for idx, uid in enumerate(sorted(unique_user_ids))}
			self._user_index = {idx: uid for uid, idx in self._user_id_map.items()}

			print(f"[LightFM] Found {len(interactions)} interactions from {len(unique_user_ids)} users")

			# Build interaction matrix
			interaction_matrix = self._build_interaction_matrix(interactions)
			
			if interaction_matrix is None or interaction_matrix.nnz == 0:
				print("[LightFM] Empty interaction matrix, using bootstrap")
				self._train_bootstrap_model()
				return

			# Train model with real data
			self._train_model(interaction_matrix)
			self._last_trained = datetime.now()
			print(f"[LightFM] Model trained successfully with {interaction_matrix.nnz} interactions")

		except Exception as e:
			print(f"[LightFM] Error loading data: {e}, using bootstrap fallback")
			self._items = self._bootstrap_items()
			self._train_bootstrap_model()

	def _build_interaction_matrix(self, interactions: List[Dict]) -> Optional[csr_matrix]:
		"""Build sparse interaction matrix from MongoDB interactions"""
		if csr_matrix is None:
			return None

		try:
			n_users = len(self._user_id_map)
			n_items = len(self._items)

			if n_users == 0 or n_items == 0:
				return None

			# Build COO matrix (row, col, data)
			rows = []
			cols = []
			weights = []

			for inter in interactions:
				user_id = inter['userId']
				task_id = str(inter.get('taskId', ''))
				
				if user_id not in self._user_id_map:
					continue
				if task_id not in self._item_index:
					continue

				user_idx = self._user_id_map[user_id]
				item_idx = self._item_index[task_id]
				
				# Use interaction weight (0-10 scale, normalize to 0-1)
				weight = inter.get('interactionWeight', 1.0) / 10.0
				# Boost completed tasks
				if inter.get('completed', False):
					weight *= 1.5
				# Boost high scores
				if inter.get('score') is not None:
					score_boost = inter['score'] / 100.0
					weight *= (1.0 + score_boost)

				rows.append(user_idx)
				cols.append(item_idx)
				weights.append(min(1.0, weight))  # Cap at 1.0

			if len(rows) == 0:
				return None

			# Convert to CSR matrix
			matrix = csr_matrix((weights, (rows, cols)), shape=(n_users, n_items))
			return matrix

		except Exception as e:
			print(f"[LightFM] Error building interaction matrix: {e}")
			return None

	def _train_model(self, interaction_matrix: csr_matrix) -> None:
		"""Train LightFM model with interaction matrix"""
		if LightFM is None or interaction_matrix is None:
			self._model = None
			return

		try:
			# Use WARP loss for implicit feedback (only positive interactions)
			model = LightFM(
				no_components=32,  # Increased for better learning
				loss='warp',
				learning_rate=0.05,
				item_alpha=1e-6,
				user_alpha=1e-6,
				random_state=42
			)

			# Train model
			model.fit(interaction_matrix, epochs=30, num_threads=2, verbose=False)
			self._model = model
			print(f"[LightFM] Model trained: {interaction_matrix.shape[0]} users, {interaction_matrix.shape[1]} items")

		except Exception as e:
			print(f"[LightFM] Error training model: {e}")
			self._model = None

	def _bootstrap_items(self) -> List[Dict]:
		"""Fallback catalog if no database connection"""
		catalog = [
			{'id': 'easy_1', 'title': 'Read a short article on JS basics', 'difficulty': 'easy', 'minutes': 20},
			{'id': 'easy_2', 'title': 'Practice 5 array methods', 'difficulty': 'easy', 'minutes': 25},
			{'id': 'medium_1', 'title': 'Build a small TODO app', 'difficulty': 'medium', 'minutes': 60},
			{'id': 'medium_2', 'title': 'Solve 3 algorithm problems', 'difficulty': 'medium', 'minutes': 45},
			{'id': 'hard_1', 'title': 'Refactor a component with tests', 'difficulty': 'hard', 'minutes': 90},
			{'id': 'hard_2', 'title': 'Implement auth flow end-to-end', 'difficulty': 'hard', 'minutes': 120},
		]
		return catalog

	def _train_bootstrap_model(self) -> None:
		"""Train simple bootstrap model with prototype users"""
		if LightFM is None:
			self._model = None
			return

		# Rebuild item index
		self._item_index = {item['id']: idx for idx, item in enumerate(self._items)}

		n_users = 3  # new, intermediate, expert
		n_items = len(self._items)

		# Build a tiny implicit interactions matrix
		interactions = np.zeros((n_users, n_items), dtype=np.float32)

		# Heuristics: map profiles to preferences over difficulties
		# user 0: new -> prefers easy, some medium
		interactions[0, [self._item_index.get('easy_1', 0), self._item_index.get('easy_2', 1)]] = 1.0
		if 'medium_1' in self._item_index:
			interactions[0, [self._item_index['medium_1']]] = 0.5
		# user 1: intermediate -> prefers medium, some hard
		if 'medium_1' in self._item_index and 'medium_2' in self._item_index:
			interactions[1, [self._item_index['medium_1'], self._item_index['medium_2']]] = 1.0
		if 'hard_1' in self._item_index:
			interactions[1, [self._item_index['hard_1']]] = 0.6
		# user 2: expert -> prefers hard, some medium
		if 'hard_1' in self._item_index and 'hard_2' in self._item_index:
			interactions[2, [self._item_index['hard_1'], self._item_index['hard_2']]] = 1.0
		if 'medium_2' in self._item_index:
			interactions[2, [self._item_index['medium_2']]] = 0.5

		# Simple LightFM model; use WARP for ranking
		model = LightFM(no_components=16, loss='warp') if LightFM is not None else None
		if model is None:
			self._model = None
			return

		# Convert to CSR required by LightFM
		mat = csr_matrix(interactions)
		model.fit(mat, epochs=20, num_threads=1, verbose=False)
		self._model = model

	def _profile_to_user_idx(self, profile: UserProfile) -> Optional[int]:
		"""Map user profile to user index in model"""
		# If we have real user data, try to find the user
		if profile.user_id in self._user_id_map:
			return self._user_id_map[profile.user_id]
		
		# Otherwise, map to prototype user based on experience
		if profile.experience == 'new':
			return 0
		if profile.experience == 'intermediate':
			return 1
		return 2

	def ingest_behavior_event(self, user_id: str, task_id: str, completed: bool, 
	                         time_spent: float, difficulty: str, interaction_weight: float = 1.0) -> None:
		"""Store behavior event and optionally trigger retraining"""
		# Note: Actual storage happens in Next.js API (UserTaskInteraction model)
		# This method is kept for backwards compatibility
		# The model will pick up new interactions on next recommendation request
		# For incremental updates, we could implement a cache invalidation mechanism here
		pass

	def recommend(self, user_profile: UserProfile) -> Dict:
		"""Generate recommendations using LightFM or fallback rules"""
		scores = []
		
		if self._model is not None and len(self._items) > 0:
			user_idx = self._profile_to_user_idx(user_profile)
			
			if user_idx is not None:
				try:
					# Predict scores for all items
					n_items = len(self._items)
					item_ids = np.arange(n_items, dtype=np.int32)
					pred = self._model.predict(
						user_ids=user_idx, 
						item_ids=item_ids, 
						num_threads=1
					)
					scores = list(enumerate(pred.tolist()))
				except Exception as e:
					print(f"[LightFM] Prediction error: {e}, using fallback")
					scores = []
		
		# Fallback to rule-based if no scores
		if not scores:
			pref = {'easy': 0.0, 'medium': 0.0, 'hard': 0.0}
			if user_profile.experience == 'new':
				pref = {'easy': 1.0, 'medium': 0.5, 'hard': 0.1}
			elif user_profile.experience == 'intermediate':
				pref = {'easy': 0.3, 'medium': 1.0, 'hard': 0.6}
			else:
				pref = {'easy': 0.2, 'medium': 0.6, 'hard': 1.0}
			
			scores = [
				(i, pref.get(self._items[i].get('difficulty', 'medium'), 0.5)) 
				for i in range(len(self._items))
			]

		# Sort by score and get top recommendations
		sorted_idx = sorted(scores, key=lambda x: x[1], reverse=True)
		top = [self._items[i] for i, _ in sorted_idx[:5]]

		# Calculate tasks per week based on profile
		tasks_per_week = max(2, min(7, int(round(
			user_profile.time_spent_coding / 3 + user_profile.skill_level / 2
		))))

		# Determine difficulty from top recommendation
		difficulty = 'medium'
		if top:
			difficulty = top[0].get('difficulty', 'medium')

		return {
			'tasks': [{
				'title': it['title'], 
				'minutes': it.get('minutes', 30), 
				'details': it.get('difficulty', 'medium'),
				'category': it.get('category', 'general'),
				'skills': it.get('skills', [])
			} for it in top],
			'tasksPerWeek': tasks_per_week,
			'difficulty': difficulty,
			'adaptiveMessage': self._adaptive_message(user_profile, difficulty, tasks_per_week),
			'userProfile': user_profile.to_dict(),
			'recommendationTimestamp': datetime.now().isoformat(),
			'modelType': 'lightfm' if self._model is not None else 'rule-based',
			'taskCount': len(self._items)
		}

	def generate_quiz(self, user_profile: UserProfile) -> dict:
		"""Generate a quiz personalized to the user's profile"""
		topic = 'basics'
		if user_profile.experience == 'intermediate':
			topic = 'arrays'
		elif user_profile.experience == 'expert':
			topic = 'algorithms'
		
		quiz_bank = {
			'basics': [
				{
					'question': 'What is the output of console.log(typeof null)?',
					'choices': ['object', 'null', 'undefined', 'number'],
					'answer': 'object',
					'explanation': 'typeof null is a weird JS quirk: it returns "object".'
				},
				{
					'question': 'Which method creates a new array with the results of calling a function on every element?',
					'choices': ['map', 'filter', 'reduce', 'forEach'],
					'answer': 'map',
					'explanation': 'Array.map() maps one array to another.'
				}
			],
			'arrays': [
				{
					'question': 'How do you remove the last item from an array in JS?',
					'choices': ['pop()', 'push()', 'shift()', 'splice()'],
					'answer': 'pop()',
					'explanation': 'pop() removes the last element.'
				},
				{
					'question': 'What does filter() return if no elements match?',
					'choices': ['[]', 'null', 'undefined', 'false'],
					'answer': '[]',
					'explanation': 'Always returns an array.'
				}
			],
			'algorithms': [
				{
					'question': 'Which sorting algorithm has worst-case O(n^2) time?',
					'choices': ['Bubble sort', 'Merge sort', 'Quick sort', 'Heap sort'],
					'answer': 'Bubble sort',
					'explanation': 'Bubble sort is O(n^2) in worst case.'
				},
				{
					'question': 'What does binary search require?',
					'choices': ['Sorted array', 'Unsorted array', 'Objects', 'Numbers only'],
					'answer': 'Sorted array',
					'explanation': 'Binary search only works on sorted arrays.'
				}
			]
		}
		
		questions = quiz_bank.get(topic, quiz_bank['basics'])[:2]

		return {
			'topic': topic,
			'questions': [
				{
					'text': q['question'],
					'choices': q['choices'],
					'explanation': q['explanation'],
					'correct': q['answer']
				} for q in questions
			],
			'userProfile': user_profile.to_dict(),
			'generatedAt': datetime.now().isoformat(),
		}

	def _adaptive_message(self, profile: UserProfile, difficulty: str, tpw: int) -> str:
		base = f"Based on your profile, we'll target {difficulty} tasks at {tpw}/week."
		if profile.motivation == 'high':
			return base + " Keep up the momentum!"
		if profile.motivation == 'low':
			return base + " We'll start gentle and ramp up."
		return base

	def retrain(self) -> bool:
		"""Manually trigger model retraining with latest data"""
		try:
			print("[LightFM] Manual retraining triggered")
			self._load_data_and_train()
			return self._model is not None
		except Exception as e:
			print(f"[LightFM] Retraining failed: {e}")
			return False
