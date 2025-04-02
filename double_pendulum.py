import pygame
import math
import sys
from collections import deque
import random

# --- Constants ---
WIDTH, HEIGHT = 900, 700
WHITE = pygame.Color(255, 255, 255)
BLACK = pygame.Color(0, 0, 0)
RED = pygame.Color(255, 0, 0)
DARK_GREY = pygame.Color(50, 50, 50)
ORANGE = pygame.Color(255, 165, 0)
FPS = 60

# --- Simulation Parameters ---
G = 9.81
PIXELS_PER_METER = 300
DT = 1.0 / (FPS * 8) # Keep stability settings
N_STEPS_PER_FRAME = 8
DAMPING = 0.1
INITIAL_OMEGA_RANGE = 0.5

# --- Trail Parameters ---
TRAIL_DURATION_MS = 1200
INTERPOLATION_STEPS_PER_SEGMENT = 8

# --- Pendulum Definitions ---
NUM_PENDULUMS = 2
pendulums = []
p1_params = { # ... (parameters same as before) ...
    "id": 0, "pivot": (WIDTH * 0.4, HEIGHT // 3), "L1_display": 150.0, "L2_display": 150.0, "M1": 1.0, "M2": 1.0, "theta1_init": math.pi / 1.5, "theta2_init": math.pi / 1.0, "omega1_init": 0.0, "omega2_init": 0.0, "bob1_color": BLACK, "bob2_color": RED, "trail_color": pygame.Color(0, 0, 200), "trail_maxlen_factor": 2.0
}
p2_params = { # ... (parameters same as before) ...
    "id": 1, "pivot": (WIDTH * 0.6, HEIGHT // 3), "L1_display": 100.0, "L2_display": 100.0, "M1": 1.0, "M2": 1.0, "theta1_init": math.pi / 1.8, "theta2_init": math.pi / 1.5, "omega1_init": 0.0, "omega2_init": 0.0, "bob1_color": DARK_GREY, "bob2_color": ORANGE, "trail_color": pygame.Color(180, 80, 0), "trail_maxlen_factor": 1.5
}

# --- Initialize Pendulum States ---
for params in [p1_params, p2_params]:
    params["L1_phys"] = params["L1_display"] / PIXELS_PER_METER
    params["L2_phys"] = params["L2_display"] / PIXELS_PER_METER
    params["theta1"] = params["theta1_init"]; params["theta2"] = params["theta2_init"]
    params["omega1"] = params["omega1_init"]; params["omega2"] = params["omega2_init"]
    est_maxlen = int(TRAIL_DURATION_MS / (DT * 1000 * N_STEPS_PER_FRAME)) * params["trail_maxlen_factor"] if DT > 0 else 200
    params["trail"] = deque(maxlen=max(1, int(est_maxlen)))
    pendulums.append(params)

# --- Pygame Setup ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Two Double Pendulums - Input Fix Attempt")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# --- Helper & Physics Functions ---

def catmull_rom_point(P0, P1, P2, P3, t):
    t2=t*t; t3=t2*t; c0=P1; c1=(-0.5*P0+0.5*P2); c2=(P0-2.5*P1+2.0*P2-0.5*P3); c3=(-0.5*P0+1.5*P1-1.5*P2+0.5*P3)
    return c0+c1*t+c2*t2+c3*t3

def get_accelerations(t1, t2, w1, w2, l1, l2, m1, m2):
    g=G; den=l1*l2*(m1+m2*math.sin(t1-t2)**2);
    if abs(den)<1e-12: den=1e-12
    n1a = -m2*l1*w1**2*math.sin(t1-t2)*math.cos(t1-t2); n1b = -m2*l2*w2**2*math.sin(t1-t2)
    n1c = -(m1+m2)*g*l1*math.sin(t1); n1d = m2*g*l2*math.sin(t2)*math.cos(t1-t2)
    al1 = (n1a+n1b+n1c+n1d)/den
    n2a = (m1+m2)*l1*w1**2*math.sin(t1-t2); n2b = -(m1+m2)*g*math.sin(t2)
    n2c = (m1+m2)*g*math.sin(t1)*math.cos(t1-t2); n2d = m2*l2*w2**2*math.sin(t1-t2)*math.cos(t1-t2)
    al2 = (n2a+n2b+n2c+n2d)/den
    al1 -= DAMPING * w1; al2 -= DAMPING * w2
    return al1, al2

# *** Moved derivative calculator OUTSIDE main loop ***
# It takes the pendulum 'p' dictionary as an argument now
def pendulum_derivative_calculator(state, p):
    th1, th2, om1, om2 = state
    al1, al2 = get_accelerations(th1, th2, om1, om2,
                                 p['L1_phys'], p['L2_phys'],
                                 p['M1'], p['M2'])
    # Check for bad acceleration values (still useful)
    if math.isnan(al1) or math.isinf(al1) or math.isnan(al2) or math.isinf(al2):
        # Print only once if possible, or remove if flooding terminal
        # print(f"Warning: Unstable accelerations detected for pendulum {p['id']}")
        return [om1, om2, 0.0, 0.0] # Return non-fatal values
    return [om1, om2, al1, al2]

# *** rk4_step now takes the calculator function AND pendulum 'p' ***
def rk4_step(t1, t2, w1, w2, dt, p, deriv_calculator):
    y = [t1, t2, w1, w2]
    # Define the state-only function needed by RK4 steps, capturing 'p'
    def f(state):
        return deriv_calculator(state, p)

    # Internal NaN/Inf checks remain useful
    k1 = [dt*val for val in f(y)]
    if any(math.isnan(v) or math.isinf(v) for v in k1): return t1, t2, w1, w2
    y2 = [y[i]+0.5*k1[i] for i in range(4)]; k2 = [dt*val for val in f(y2)]
    if any(math.isnan(v) or math.isinf(v) for v in k2): return t1, t2, w1, w2
    y3 = [y[i]+0.5*k2[i] for i in range(4)]; k3 = [dt*val for val in f(y3)]
    if any(math.isnan(v) or math.isinf(v) for v in k3): return t1, t2, w1, w2
    y4 = [y[i]+k3[i] for i in range(4)]; k4 = [dt*val for val in f(y4)]
    if any(math.isnan(v) or math.isinf(v) for v in k4): return t1, t2, w1, w2
    yn = [y[i]+(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6.0 for i in range(4)]
    if any(math.isnan(v) or math.isinf(v) for v in yn): return t1, t2, w1, w2
    return yn[0], yn[1], yn[2], yn[3]

# --- Main Loop ---
running = True; paused = False
last_time_ms = pygame.time.get_ticks()

while running:
    current_time_ms = pygame.time.get_ticks() # Update time each frame

    # --- Event Handling (Check FIRST) ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE: running = False
            if event.key == pygame.K_SPACE: paused = not paused
            if event.key == pygame.K_r:
                for p in pendulums:
                    angle_range = math.pi * 0.95
                    p["theta1"] = random.uniform(-angle_range, angle_range)
                    p["theta2"] = random.uniform(-angle_range, angle_range)
                    p["omega1"] = random.uniform(-INITIAL_OMEGA_RANGE, INITIAL_OMEGA_RANGE)
                    p["omega2"] = random.uniform(-INITIAL_OMEGA_RANGE, INITIAL_OMEGA_RANGE)
                    p["trail"].clear()

    # --- Update Physics (if not paused) ---
    if not paused:
        for p in pendulums: # Iterate through each pendulum
            # Perform physics steps using the external calculator function
            for _ in range(N_STEPS_PER_FRAME):
                 # Pass pendulum 'p' and the calculator function to rk4_step
                 th1_new, th2_new, om1_new, om2_new = \
                    rk4_step(p["theta1"], p["theta2"], p["omega1"], p["omega2"],
                             DT, p, pendulum_derivative_calculator)

                 # Simple update: Assume rk4 handled internal NaNs by returning old state
                 # We removed the extra 'unstable_step' check here for now
                 p["theta1"], p["theta2"], p["omega1"], p["omega2"] = th1_new, th2_new, om1_new, om2_new


            # Add point to trail (only if simulation is running)
            px, py = p["pivot"]
            x1_d = px + p["L1_display"] * math.sin(p["theta1"])
            y1_d = py + p["L1_display"] * math.cos(p["theta1"])
            x2_d = x1_d + p["L2_display"] * math.sin(p["theta2"])
            y2_d = y1_d + p["L2_display"] * math.cos(p["theta2"])
            # Only add if coords are valid (basic check)
            if not (math.isnan(x2_d) or math.isinf(x2_d) or math.isnan(y2_d) or math.isinf(y2_d)):
                 p["trail"].append((pygame.math.Vector2(x2_d, y2_d), current_time_ms))


    # --- Trail Cleanup (Run regardless of pause) ---
    for p in pendulums:
        while p["trail"] and current_time_ms - p["trail"][0][1] > TRAIL_DURATION_MS:
            p["trail"].popleft()

    # --- Drawing (Run regardless of pause) ---
    screen.fill(WHITE)
    # Draw Trails
    for p in pendulums:
        trail_data=p["trail"]; trail_color=p["trail_color"]
        if len(trail_data) >= 2:
            t_pos = [pd[0] for pd in trail_data]; ntp = len(t_pos)
            # Check if trail positions are valid before drawing
            if not any(math.isnan(v.x) or math.isinf(v.x) or math.isnan(v.y) or math.isinf(v.y) for v in t_pos):
                for i in range(ntp - 1):
                    P1=t_pos[i]; P2=t_pos[i+1]; P0=t_pos[i-1] if i>0 else P1; P3=t_pos[i+2] if i<ntp-2 else P2
                    time1=trail_data[i][1]; age_ms=current_time_ms-time1; fade=min(1.0,max(0.0,age_ms/TRAIL_DURATION_MS))
                    try: scol=trail_color.lerp(WHITE,fade)
                    except (AttributeError, ValueError): scol=trail_color # Handle potential lerp errors
                    inter_pts=[]
                    for j in range(INTERPOLATION_STEPS_PER_SEGMENT + 1):
                        t=j/INTERPOLATION_STEPS_PER_SEGMENT; pt=catmull_rom_point(P0,P1,P2,P3,t)
                        # Check interpolated point too
                        if math.isnan(pt.x) or math.isinf(pt.x) or math.isnan(pt.y) or math.isinf(pt.y):
                             inter_pts = [] # Clear points if instability occurs during interpolation
                             break
                        inter_pts.append(pt)
                    if len(inter_pts)>=2:
                        try:
                             draw_pts=[(int(pt.x), int(pt.y)) for pt in inter_pts]; pygame.draw.aalines(screen,scol,False,draw_pts)
                        except OverflowError: # Catch potential errors converting huge floats to int
                             pass # Skip drawing this segment if coords are too large

    # Draw Pendulums
    for p in pendulums:
        px,py=p["pivot"]; x1d=px+p["L1_display"]*math.sin(p["theta1"]); y1d=py+p["L1_display"]*math.cos(p["theta1"])
        x2d=x1d+p["L2_display"]*math.sin(p["theta2"]); y2d=y1d+p["L2_display"]*math.cos(p["theta2"])
        # Check coordinates before drawing
        if not any(math.isnan(v) or math.isinf(v) for v in [x1d, y1d, x2d, y2d]):
             try:
                 pygame.draw.line(screen,BLACK,(px,py),(int(x1d),int(y1d)),2)
                 pygame.draw.line(screen,BLACK,(int(x1d),int(y1d)),(int(x2d),int(y2d)),2)
                 bob1r=int(p["M1"]*6+p["L1_display"]*0.02); bob2r=int(p["M2"]*6+p["L2_display"]*0.02)
                 pygame.draw.circle(screen,p["bob1_color"],(int(x1d),int(y1d)),bob1r)
                 pygame.draw.circle(screen,p["bob2_color"],(int(x2d),int(y2d)),bob2r)
             except OverflowError:
                 pass # Skip drawing if coords too large for int conversion

    # Draw Text
    tcol=BLACK if not paused else RED; stat_txt="Space: Pause/Resume | R: Reset" if not paused else "PAUSED (Space: Resume | R: Reset)"
    tsurf=font.render(stat_txt,True,tcol); screen.blit(tsurf,(10,10))
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()