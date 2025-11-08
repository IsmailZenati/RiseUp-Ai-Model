from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime

try:
	from lightfm import LightFM
except Exception:  # allow runtime without LightFM in dev
	LightFM = None  # type: ignore


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

		self._items = self._bootstrap_items()
		self._item_index = {item['id']: idx for idx, item in enumerate(self._items)}
		self._model: Optional[LightFM] = None

		if LightFM is not None:
			self._train_bootstrap_model()

	def _bootstrap_items(self) -> List[Dict]:
		# Minimal shared catalog so LightFM has shared items to rank
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
		if LightFM is None:
			self._model = None
			return

		n_users = 3  # new, intermediate, expert
		n_items = len(self._items)

		# Build a tiny implicit interactions matrix
		interactions = np.zeros((n_users, n_items), dtype=np.float32)

		# Heuristics: map profiles to preferences over difficulties
		# user 0: new -> prefers easy, some medium
		interactions[0, [self._item_index['easy_1'], self._item_index['easy_2']]] = 1.0
		interactions[0, [self._item_index['medium_1']]] = 0.5
		# user 1: intermediate -> prefers medium, some hard
		interactions[1, [self._item_index['medium_1'], self._item_index['medium_2']]] = 1.0
		interactions[1, [self._item_index['hard_1']]] = 0.6
		# user 2: expert -> prefers hard, some medium
		interactions[2, [self._item_index['hard_1'], self._item_index['hard_2']]] = 1.0
		interactions[2, [self._item_index['medium_2']]] = 0.5

		# Simple LightFM model; use WARP for ranking
		model = LightFM(no_components=16, loss='warp') if LightFM is not None else None
		if model is None:
			self._model = None
			return

		# Fit: for demo, run a few epochs over dense interactions
		# Convert to CSR required by LightFM
		from scipy.sparse import csr_matrix
		mat = csr_matrix(interactions)
		model.fit(mat, epochs=20, num_threads=1)
		self._model = model

	def _profile_to_user_idx(self, profile: UserProfile) -> int:
		if profile.experience == 'new':
			return 0
		if profile.experience == 'intermediate':
			return 1
		return 2

	def ingest_behavior_event(self, user_id: str, task_id: str, completed: bool, time_spent: float, difficulty: str) -> None:
		# Placeholder: could adjust internal weights or trigger retraining
		return

	def recommend(self, user_profile: UserProfile) -> Dict:
		# Score items using LightFM if available; otherwise simple rules
		scores = []
		if self._model is not None:
			user_idx = self._profile_to_user_idx(user_profile)
			# Predict scores for all items for this prototype user index
			from scipy.sparse import csr_matrix
			n_items = len(self._items)
			# LightFM predict expects user_ids and item_ids
			item_ids = np.arange(n_items, dtype=np.int32)
			pred = self._model.predict(user_ids=user_idx, item_ids=item_ids, num_threads=1)
			scores = list(enumerate(pred.tolist()))
		else:
			# Rule fallback by difficulty
			pref = {'easy': 0.0, 'medium': 0.0, 'hard': 0.0}
			if user_profile.experience == 'new':
				pref = {'easy': 1.0, 'medium': 0.5, 'hard': 0.1}
			elif user_profile.experience == 'intermediate':
				pref = {'easy': 0.3, 'medium': 1.0, 'hard': 0.6}
			else:
				pref = {'easy': 0.2, 'medium': 0.6, 'hard': 1.0}
			scores = [(i, pref[self._items[i]['difficulty']]) for i in range(len(self._items))]

		sorted_idx = sorted(scores, key=lambda x: x[1], reverse=True)
		top = [self._items[i] for i, _ in sorted_idx[:5]]

		tasks_per_week = max(2, min(7, int(round(user_profile.time_spent_coding / 3 + user_profile.skill_level / 2))))
		difficulty = 'easy' if top[0]['difficulty'] == 'easy' else ('hard' if top[0]['difficulty'] == 'hard' else 'medium')

		return {
			'tasks': [{'title': it['title'], 'minutes': it['minutes'], 'details': it['difficulty']} for it in top],
			'tasksPerWeek': tasks_per_week,
			'difficulty': difficulty,
			'adaptiveMessage': self._adaptive_message(user_profile, difficulty, tasks_per_week),
			'userProfile': user_profile.to_dict(),
			'recommendationTimestamp': datetime.now().isoformat(),
		}

	def generate_quiz(self, user_profile: UserProfile) -> dict:
		"""
		Generate a quiz personalized to the user's profile.
		This demo picks questions based on skill level/experience, with room for LightFM/NLP in future.
		"""
		# Example topics based on LightFM or profile
		# For now, map experience to topic; use LightFM for more advanced selection in the future.
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
		# Select topic questions
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
