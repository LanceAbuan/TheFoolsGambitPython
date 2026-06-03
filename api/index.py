"""Vercel Python serverless function — thin wrapper around game engine."""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "TheFoolsGambitPython", "backend"))
from game import new_game, make_move, ai_move, undo  # noqa: E402


def handler(request):
    """Handle Vercel request and route to appropriate game function."""
    try:
        body = json.loads(request.body.decode("utf-8")) if request.body else {}
    except Exception:
        body = {}

    url = request.url

    if "new-game" in url:
        result = new_game(ai_depth=body.get("aiDepth", 3))
    elif "make-move" in url:
        uci = body.get("uci")
        fen = body.get("fen")
        legal = body.get("legal")
        if not uci or not fen:
            return {"statusCode": 400, "body": json.dumps({"error": "Missing uci or fen"})}
        result = make_move(fen, uci, legal=legal)
    elif "ai-move" in url:
        fen = body.get("fen")
        legal = body.get("legal")
        ai_depth = body.get("aiDepth", 3)
        if not fen:
            return {"statusCode": 400, "body": json.dumps({"error": "Missing fen"})}
        result = ai_move(fen, legal=legal, ai_depth=ai_depth)
    elif "undo" in url:
        fen = body.get("fen")
        if not fen:
            return {"statusCode": 400, "body": json.dumps({"error": "Missing fen"})}
        result = undo(fen)
    else:
        result = new_game()

    return {"statusCode": 200, "body": json.dumps(result)}
