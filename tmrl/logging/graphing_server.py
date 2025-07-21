import asyncio
import websockets
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import defaultdict, deque
import threading

PORT = 6789
MAX_POINTS = 100
VARIABLES = ["speed", "distance", "displacement", "gas", "braking", "input steer", "gear", "rpm", "reward", "collision"]

data = defaultdict(lambda: deque(maxlen=MAX_POINTS))
last_data = defaultdict(lambda: None)

fig, axes = plt.subplots(len(VARIABLES), 1, figsize=(8, 8), sharex=True)
if len(VARIABLES) == 1:
    axes = [axes]
lines = []

for ax, var in zip(axes, VARIABLES):
    line, = ax.plot([], [], label=var)
    lines.append(line)
    ax.set_ylabel(var)
axes[-1].set_xlabel("Time step")

annotation_box = fig.text(0.82, 0.5, '', transform=fig.transFigure, va='center', ha='left',
                          bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black"))

def update_plot(frame):
    update_needed = False
    for var in VARIABLES:
        current_data = list(data[var])
        if current_data and current_data[-1] != last_data[var]:
            update_needed = True
            last_data[var] = current_data[-1]
            break

    if update_needed:
        for line, var in zip(lines, VARIABLES):
            y = list(data[var])
            x = list(range(len(y)))
            line.set_data(x, y)
            line.axes.relim()
            line.axes.autoscale_view()
    return lines

def on_mouse_move(event):
    if not event.inaxes:
        annotation_box.set_text('')
        return

    try:
        index = int(round(event.xdata))
    except:
        annotation_box.set_text('')
        return

    texts = [f"x = {index}"]
    for var in VARIABLES:
        y_list = list(data[var])
        if 0 <= index < len(y_list):
            texts.append(f"{var}: {y_list[index]:.3f}")
        else:
            texts.append(f"{var}: N/A")

    annotation_box.set_text('\n'.join(texts))
    vline.set_xdata(index)
    fig.canvas.draw_idle()

def on_data_update(frame):
    if fig.canvas.manager.get_window().GetActiveWindow() and plt.get_current_fig_manager().canvas.widgetlock.locked():
        event = fig.canvas.manager.get_window().GetEvent()
        if event and event.inaxes:
            on_mouse_move(event)
    return []

fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

async def handle_client(ws):
    print("[Graphing Server] Client connected")
    async for msg in ws:
        try:
            payload = json.loads(msg)
            for key in VARIABLES:
                if key in payload:
                    data[key].append(payload[key])
        except Exception as e:
            print("Error parsing JSON:", e)

async def start_ws_server():
    print(f"[Graphing Server] Starting WebSocket server on ws://127.0.0.1:{PORT}")
    async with websockets.serve(handle_client, "127.0.0.1", PORT):
        await asyncio.Future()

def start_asyncio_loop():
    asyncio.run(start_ws_server())

if __name__ == "__main__":
    threading.Thread(target=start_asyncio_loop, daemon=True).start()

    try:
        manager = plt.get_current_fig_manager()
        manager.window.wm_geometry("+1350+0")
    except Exception as e:
        print(f"[Graphing Server] Could not set window position: {e}")

    ani = FuncAnimation(fig, update_plot, interval=100)
    ani_data = FuncAnimation(fig, on_data_update, interval=100)
    plt.tight_layout()
    plt.show()