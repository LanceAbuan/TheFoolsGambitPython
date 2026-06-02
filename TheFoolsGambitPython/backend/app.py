from typing import Optional
from flask import Flask, render_template, request, jsonify
from game import ChessGame
from ai import ai

app = Flask(__name__)

games = {}
active_game_id = None


def get_game() -> Optional[ChessGame]:
    return games.get(active_game_id)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/new-game', methods=['POST'])
def new_game():
    global active_game_id
    game_id = f"game_{len(games) + 1}"
    games[game_id] = ChessGame()
    active_game_id = game_id
    return jsonify(games[game_id].get_state())


@app.route('/api/move', methods=['POST'])
def make_move():
    game = get_game()
    if game is None:
        return jsonify({"error": "No active game"}), 400

    move_uci = request.json.get('move')
    if not move_uci:
        return jsonify({"error": "No move provided"}), 400

    return jsonify(game.make_move(move_uci))


@app.route('/api/undo', methods=['POST'])
def undo_move():
    game = get_game()
    if game is None:
        return jsonify({"error": "No active game"}), 400

    return jsonify(game.undo_move())


@app.route('/api/state')
def get_state():
    game = get_game()
    if game is None:
        return jsonify({"error": "No active game"}), 400

    s = game.get_state()
    s['pgn'] = game.get_pgn()
    return jsonify(s)


@app.route('/api/ai-move', methods=['POST'])
def ai_move():
    game = get_game()
    if game is None:
        return jsonify({"error": "No active game"}), 400

    move = ai.generate(game.board)
    if not move:
        return jsonify({"error": "No legal moves for AI"}), 400

    return jsonify(game.make_move(move))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
