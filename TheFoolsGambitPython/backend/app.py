from flask import Flask, render_template, jsonify, request
from game import game_manager
from ai import ai

app = Flask(__name__, static_folder="static", template_folder="templates")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/new-game", methods=["POST"])
def new_game():
    """Start a fresh game. Accepts optional { 'aiDepth': 3 }."""
    data = request.get_json(silent=True) or {}
    ai_depth = data.get("aiDepth", 3)
    ai.depth = ai_depth
    result = game_manager.new_game()
    return jsonify(result)


@app.route("/api/make-move", methods=["POST"])
def make_move():
    """Submit a move as UCI string (e.g. 'e2e4'). Returns updated state."""
    data = request.get_json(silent=True) or {}
    uci = data.get("uci")
    if not uci:
        return jsonify({"error": "Missing uci"}), 400

    result = game_manager.make_move(uci)
    return jsonify(result)


@app.route("/api/ai-move", methods=["POST"])
def ai_move():
    """Request AI to play. Returns updated state."""
    result = game_manager.ai_move()
    return jsonify(result)


@app.route("/api/state")
def state():
    """Get current game state."""
    return jsonify(game_manager.state())


@app.route("/api/undo", methods=["POST"])
def undo():
    """Undo the last move (one ply)."""
    result = game_manager.undo()
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
