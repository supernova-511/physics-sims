# run this file (pendulum.py) from the Anaconda prompt, by typing the command 'python pendulum.py'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
plt.style.use("dark_background")

m = 1
g = 10
L = 0.5

def pendulum(t, S):
    """S = [x, xDot]"""
    rate0 = S[1]
    rate1 = -g * np.sin(S[0] / L)
    return [rate0, rate1]

dt = 0.005
T = 5

sol = solve_ivp(pendulum, t_span=(0, T), y0=[np.pi/6, 0], t_eval=np.arange(0, T, dt), method="DOP853")

X = L * np.sin(sol.y[0]/L)
Y = -L * np.cos(sol.y[0]/L)
xDot = sol.y[1]
Ke = (1/2) * m * (xDot**2)
Pe = m*g*(L + Y)
H = Ke + Pe

fig, (ax1, ax2) =  plt.subplots(1, 2, figsize=(8, 8))

ax1.set_xlim(-L-0.1, L+0.1)
ax1.set_ylim(-L-0.1, L+0.1)
ax1.set_aspect('equal')
ax1.axis('off')

ax2.set_xlim(-np.pi/2, np.pi/2)
ax2.set_ylim(0, 1*m*g*L+0.1)
ax2.set_xlabel(r"$\theta$")
ax2.set_ylabel(r"$V(\theta)$")
x1 = np.arange(-np.pi/2, np.pi/2, 0.01)
y1 = m*g*L*(1 - np.cos(x1))

ax2.plot(x1, y1, color="#ff4fbe", label=r"$V$")
K_arrow, = ax2.plot([], [], '-', lw=2, color="#4f84ff", label=r"$T$")
ax2.plot(x1, H[0]*np.ones(len(x1)), "--", color="#7234bf", label=r"$H$")

ax2.legend(loc="upper left")

line, = ax1.plot([], [], 'o-', lw=2, color="#7234bf")
trace, = ax1.plot([], [], '.-', lw=1, ms=2, color="#f2e338")

time_template = 'time = %.2f'
time_text = ax1.text(0.75, 0.9, '', transform=ax1.transAxes, color="#f2e338")

ke_template = 'T = %.2f'
ke_text = ax1.text(0.05, 0.90, '', transform=ax1.transAxes, color="#4f84ff")

pe_template = 'V = %.2f'
pe_text = ax1.text(0.05, 0.85, '', transform=ax1.transAxes, color="#ff4fbe")

H_template = 'H = %.2f'
He_text = ax1.text(0.05, 0.80, '', transform=ax1.transAxes, color="#7234bf")

def animate(i):
    thisx = [0, X[i]]
    thisy = [0, Y[i]]

    history_x = X[:i][-50:]
    history_y = Y[:i][-50:]

    line.set_data(thisx, thisy)
    trace.set_data(history_x, history_y)
    K_arrow.set_data([sol.y[0][i]/L, sol.y[0][i]/L], [Pe[i], H[0]])

    time_text.set_text(time_template % (i*dt))
    ke_text.set_text(ke_template % (Ke[i]))
    pe_text.set_text(pe_template % (Pe[i]))
    He_text.set_text(H_template % (H[i]))

    return trace, line, time_text, ke_text, pe_text, He_text, K_arrow

ani = animation.FuncAnimation(fig, animate, len(sol.t), interval=dt*1000, blit=True)
plt.show()
"""progress_callback = lambda i, n: print(f'Saving frame {i}/{n}')
ani.save(filename="pendulum.mp4", fps=60, dpi=300, progress_callback=progress_callback, bitrate=5000)"""
