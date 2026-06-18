"""Flask application — serves frontend and stateless chess API."""
import sys
import os
from flask import Flask, render_template, jsonify, request
from game import new_game, make_move, ai_move, undo  # noqa: E402

app = Flask(__name__, static_folder="static", template_folder="templates")

# Register training blueprint if torch is available
try:
    import torch  # noqa: F401
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from training.server import training_bp  # noqa: E402
    app.register_blueprint(training_bp)
except ImportError:
    pass


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/new-game", methods=["POST"])
def api_new_game():
    """Start a fresh game.

    Request body (optional): { "aiDepth": 3 }
    """
    data = request.get_json(silent=True) or {}
    ai_depth = data.get("aiDepth", 3)
    result = new_game(ai_depth=ai_depth)
    return jsonify(result)


@app.route("/api/make-move", methods=["POST"])
def api_make_move():
    """Apply a player move.

    Request body: { "uci": "e2e4", "fen": "...", "legal": [...] }
    """
    data = request.get_json(silent=True) or {}
    uci = data.get("uci")
    fen = data.get("fen")
    legal = data.get("legal")

    if not uci:
        return jsonify({"error": "Missing uci"}), 400
    if not fen:
        return jsonify({"error": "Missing fen"}), 400

    result = make_move(fen, uci, legal=legal)
    return jsonify(result)


@app.route("/api/ai-move", methods=["POST"])
def api_ai_move():
    """Request AI to play.

    Request body: { "fen": "...", "legal": [...], "aiDepth": 3 }
    """
    data = request.get_json(silent=True) or {}
    fen = data.get("fen")
    legal = data.get("legal")
    ai_depth = data.get("aiDepth", 3)

    if not fen:
        return jsonify({"error": "Missing fen"}), 400

    result = ai_move(fen, legal=legal, ai_depth=ai_depth)
    return jsonify(result)


@app.route("/api/undo", methods=["POST"])
def api_undo():
    """Undo last moves.

    Request body: { "fen": "<previous fen to revert to>" }
    """
    data = request.get_json(silent=True) or {}
    fen = data.get("fen")

    if not fen:
        return jsonify({"error": "Missing fen"}), 400

    result = undo(fen)
    return jsonify(result)


@app.route("/api/state")
def api_state():
    """Echo back the state for the given FEN (utility endpoint)."""
    fen = request.args.get("fen")
    if not fen:
        return jsonify({"error": "Missing fen query param"}), 400
    return jsonify(new_game())


@app.route("/api/docs")
def api_docs():
    """Fetch the documentation content."""
    try:
        # Note: The path is relative to the current working directory.
        # Since the backend is in TheFoolsGambitPython/TheFoolsGambitPython/backend,
        # and the docs are in /home/lance/TheFoolsGambitPython/docs,
        # we need to handle the path correctly.
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "docs"))
        with open(os.path.join(base_path, "ARCHITECTURE.md"), "r") as f:
            content = f.read()
        return jsonify({"content": content})
    except FileNotFoundError:
        return jsonify({"error": "Documentation file not found"}), 404


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
