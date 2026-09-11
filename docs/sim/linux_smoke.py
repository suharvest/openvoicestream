"""Isaac Sim 4.5 headless render smoke test (Linux).

Proves the platform works end-to-end: app boot + sustained render +
World physics + render-step (the three things that crash on Windows
session-0/RDP but work natively on Linux via EGL/Vulkan offscreen).

Run:  /isaac-sim/python.sh /root/linux_smoke.py
Pass = SIMAPP_OK / RENDER_60_OK / WORLD_RESET_OK / WORLD_STEP_RENDER_30_OK / CLOSED_OK
"""
import os, traceback
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'  # MUST be set in-process before the import
try:
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True})
    print('SIMAPP_OK', flush=True)
    for _ in range(60):
        app.update()
    print('RENDER_60_OK', flush=True)
    from isaacsim.core.api import World
    w = World()
    w.scene.add_default_ground_plane()
    w.reset()
    print('WORLD_RESET_OK', flush=True)
    for _ in range(30):
        w.step(render=True)
    print('WORLD_STEP_RENDER_30_OK', flush=True)
    app.close()
    print('CLOSED_OK', flush=True)
except Exception as e:
    print('EXCEPTION', repr(e), flush=True)
    traceback.print_exc()
