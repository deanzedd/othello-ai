import asyncio
import json
import secrets

from websockets.asyncio.server import broadcast, serve

from othello import BLACK, WHITE, Game
from ai.ai_player import MinimaxPlayer


JOIN = {}

WATCH = {}


async def error(websocket, message):
    """
    Send an error message.

    """
    event = {
        "type": "error",
        "message": message,
    }
    await websocket.send(json.dumps(event))


def sync_state(websocket, game, connected):
    """
    Broadcast game state to all connected clients.

    """
    event = {
        "type": "play",
        "turn": game.turn,
        "boardState": game.board_state,
        "validMoves": game.get_valid_moves,
    }
    broadcast(connected, json.dumps(event))

    # Check if game over.
    if game.is_game_over:
        event = {
            "type": "win",
            "player": game.winner,
        }
        broadcast(connected, json.dumps(event))


async def play(websocket, game, player, connected, bot=None):
    """
    Receive and process moves from a player or bot.

    """
    if bot and player == bot.player:
        # If the player is a bot, let it play its move
        move = bot.play(game)
        if move is not None:
            game.play(player, move[0], move[1])

            # Broadcast the updated game state to all connected clients
            sync_state(websocket, game, connected)

            # Check if it's the bot's turn again
            if game.turn == bot.player:
                await play(websocket, game, bot.player, connected, bot=bot)
        return

    # Handle human player's moves
    async for message in websocket:
        # Parse a "play" event from the UI.
        event = json.loads(message)
        assert event["type"] == "play"

        try:
            # Play the move.
            game.play(player, event["row"], event["col"])
        except ValueError as exc:
            # Send an "error" event if the move was illegal.
            await error(websocket, str(exc))
            continue

        # Broadcast the updated game state to all connected clients
        sync_state(websocket, game, connected)
        
        # If the opponent is a bot, let the bot play its move
        if bot and game.turn == bot.player:
            await asyncio.sleep(1)
            await play(websocket, game, WHITE, connected, bot=bot)


async def start(websocket):
    """
    Handle a connection from the first player: start a new game.

    """
    # Initialize an Othello game, the set of WebSocket connections
    # receiving moves from this game, and secret access tokens.
    game = Game()
    connected = {websocket}

    join_key = secrets.token_urlsafe(12)
    JOIN[join_key] = game, connected

    watch_key = secrets.token_urlsafe(12)
    WATCH[watch_key] = game, connected

    try:
        # Send the secret access tokens to the browser of the first player,
        # where they'll be used for building "join" and "watch" links.
        event = {
            "type": "init",
            "join": join_key,
            "watch": watch_key,
            "validMoves": game.get_valid_moves,
        }
        await websocket.send(json.dumps(event))
        # Receive and process moves from the first player.
        await play(websocket, game, BLACK, connected)
    finally:
        del JOIN[join_key]
        del WATCH[watch_key]


async def start_with_bot(websocket):
    """
    Handle a connection from the first player: start a new game against an AI bot.

    """
    game = Game()
    connected = {websocket}
    bot = MinimaxPlayer(WHITE)

    watch_key = secrets.token_urlsafe(12)
    WATCH[watch_key] = game, connected

    try:
        # Send the initial game state to the player
        event = {
            "type": "init",
            "watch": watch_key,
            "validMoves": game.get_valid_moves,
        }
        await websocket.send(json.dumps(event))

        # Let the player play as BLACK
        await play(websocket, game, BLACK, connected, bot=bot)
    finally:
        del WATCH[watch_key]


async def join(websocket, join_key):
    """
    Handle a connection from the second player: join an existing game.

    """
    # Find the Othello game.
    try:
        game, connected = JOIN[join_key]
    except KeyError:
        await error(websocket, "Game not found.")
        return

    # Register to receive updates from this game.
    connected.add(websocket)
    try:
        # Send board state.
        sync_state(websocket, game, connected)
        # Receive and process moves from the second player.
        await play(websocket, game, WHITE, connected)
    finally:
        connected.remove(websocket)


async def watch(websocket, watch_key):
    """
    Handle a connection from a spectator: watch an existing game.

    """
    # Find the Othello game.
    try:
        game, connected = WATCH[watch_key]
    except KeyError:
        await error(websocket, "Game not found.")
        return

    # Register to receive updates from this game.
    connected.add(websocket)
    try:
        # Send board state.
        sync_state(websocket, game, connected)
        # Keep the connection open, but don't receive any messages.
        while True:
            await asyncio.sleep(0.1)  # Giảm CPU usage
    finally:
        connected.remove(websocket)


async def handler(websocket):
    """
    Handle a connection and dispatch it according to who is connecting.

    """
    # Receive and parse the "init" event from the UI.
    message = await websocket.recv()
    event = json.loads(message)
    assert event["type"] == "init"

    if "bot" in event:
        # Player starts a game against the AI bot
        await start_with_bot(websocket)
    elif "join" in event:
        # Second player joins an existing game.
        await join(websocket, event["join"])
    elif "watch" in event:
        # Spectator watches an existing game.
        await watch(websocket, event["watch"])
    else:
        # First player starts a new game.
        await start(websocket)


async def main():
    async with serve(handler, "0.0.0.0", 8001) as server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())