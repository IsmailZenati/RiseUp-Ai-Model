"""
LightFM-based Recommender - Flask API
Keeps the same endpoints and response schema as the previous service.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import os
from dotenv import load_dotenv

from recommender import LightFMRecommender, UserProfile

load_dotenv()

app = Flask(__name__)

CORS(app, resources={
	"/api/*": {
		"origins": [
			os.getenv("NEXT_PUBLIC_APP_URL", "http://localhost:3000"),
			os.getenv("NEXT_PUBLIC_APP_ALT_URL", "http://localhost:3001"),
		],
		"methods": ["GET", "POST", "PUT", "DELETE"],
		"allow_headers": ["Content-Type", "Authorization"],
	}
})

# Initialize recommender (lazy)
_recommender = None

def get_recommender():
	global _recommender
	if _recommender is None:
		_recommender = LightFMRecommender(
			mongo_uri=os.getenv("MONGODB_URI", ""),
			mongo_db=os.getenv("MONGODB_DB", ""),
			use_bootstrap_fallback=os.getenv("LFM_BOOTSTRAP_FALLBACK", "1") in ("1", "true", "True"),
		)
	return _recommender


@app.route('/health', methods=['GET'])
def health_check():
	return jsonify({
		'status': 'ok',
		'service': 'lightfm',
		'timestamp': datetime.now().isoformat()
	})


@app.route('/api/recommend', methods=['POST'])
def recommend_tasks():
	try:
		data = request.get_json(force=True)

		required_fields = ['userId', 'experience', 'skillLevel', 'motivation', 'activityLevel']
		for field in required_fields:
			if field not in data:
				return jsonify({'error': f'Missing required field: {field}'}), 400

		user_profile = UserProfile(
			user_id=str(data['userId']),
			experience=str(data['experience']),
			skill_level=int(data['skillLevel']),
			age=int(data.get('age', 25)),
			time_spent_coding=float(data.get('timeSpentCoding', 5)),
			motivation=str(data['motivation']),
			activity_level=str(data['activityLevel']),
			completion_rate=float(data.get('completionRate', 0.7)),
			engagement_score=float(data.get('engagementScore', 50)),
		)

		rec = get_recommender().recommend(user_profile)
		return jsonify(rec), 200
	except Exception as e:
		return jsonify({'error': str(e)}), 500


@app.route('/api/update-behavior', methods=['POST'])
def update_behavior():
	try:
		data = request.get_json(force=True)
		# Optional: use this to incrementally update interaction weights
		get_recommender().ingest_behavior_event(
			user_id=str(data.get('userId', '')),
			task_id=str(data.get('taskId', '')),
			completed=bool(data.get('taskCompleted', False)),
			time_spent=float(data.get('timeSpent', 0)),
			difficulty=str(data.get('difficulty', 'medium')),
		)
		return jsonify({'status': 'ok'}), 200
	except Exception as e:
		return jsonify({'error': str(e)}), 500


@app.route('/api/quiz', methods=['POST'])
def quiz():
    try:
        data = request.get_json(force=True)
        # Required fields (at least userId/experience/skillLevel)
        user_profile = UserProfile(
            user_id=str(data.get('userId', '')),
            experience=str(data.get('experience', 'new')),
            skill_level=int(data.get('skillLevel', 1)),
            age=int(data.get('age', 25)),
            time_spent_coding=float(data.get('timeSpentCoding', 5)),
            motivation=str(data.get('motivation', 'medium')),
            activity_level=str(data.get('activityLevel', 'active')),
            completion_rate=float(data.get('completionRate', 0.7)),
            engagement_score=float(data.get('engagementScore', 50)),
        )
        quiz = get_recommender().generate_quiz(user_profile)
        return jsonify(quiz), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
	port = int(os.getenv('PYTHON_PORT', '5000'))
	app.run(host='0.0.0.0', port=port)
