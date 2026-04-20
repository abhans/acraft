import math
import os

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

mpl.rcParams["animation.embed_limit"] = 200.0

# --- Constants & Colors ---
NAVY: str = "#223D5A"
SEAFOAM: str = "#2B7371"
OCHRE: str = "#BBA358"


def _thrustColors(thrust):
    thrust = np.clip(thrust, 0, 1)
    return plt.cm.managua(thrust)


def createTrajAnim(dExpert, outFile="animations/simulateTrajs.mp4"):
    """
    Stitches all trajectories into a single MP4 with plot resets between them.
    """
    trajs = dExpert["trajectories"]
    numTraj = len(trajs)

    # ------------------------------------------------ Data Preparation ------------------------------------------------
    lenTrajs = [len(traj) for traj in trajs]
    tFrames = sum(lenTrajs)

    fStart = np.cumsum([0] + lenTrajs[:-1])

    print(
        f"\n[INIT] Concatenating {numTraj} trajectories for a total of {tFrames} frames."
    )

    # ------------------------------------------------ Setup Figure ------------------------------------------------
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, width_ratios=[1.2, 1])

    # Trajectory Subplot
    axTraj = fig.add_subplot(gs[:, 0])
    axTraj.set_xlim(-1, 1)
    axTraj.set_ylim(-1, 1)
    axTraj.invert_yaxis()
    axTraj.grid(True, linestyle="--", alpha=0.3, zorder=0)

    # Parameter Subplots
    axVel = fig.add_subplot(gs[0, 1])
    axVel.set_title(r"Velocity ($v_X,\ v_Y$)")

    axOrient = fig.add_subplot(gs[1, 1])
    axOrient.set_title(r"Orientation ($\theta$) | Angular Velocity ($\omega$)")

    axThr = fig.add_subplot(gs[2, 1])
    axThr.set_title(r"Actions ($T_{Left}$, $T_{Right}$)")
    axThr.set_ylim(-0.05, 1.05)  # Thrust is strictly bounded

    # ------------------------------------------------ Initialize Artists ------------------------------------------------
    # These persist across the whole video, but we clear their data on resets
    lc = mcoll.LineCollection([], cmap="managua", alpha=0.7, linewidth=4)
    axTraj.add_collection(lc)
    scatter = axTraj.scatter([], [], c=[], cmap="managua", s=0, zorder=5)

    lARMS: float = 0.08
    (droneFrame,) = axTraj.plot(
        [], [], color=SEAFOAM, lw=4, solid_capstyle="round", zorder=10
    )
    (lRotor,) = axTraj.plot(
        [], [], "o", markersize=10, markeredgecolor="white", zorder=11
    )
    (rRotor,) = axTraj.plot(
        [], [], "o", markersize=10, markeredgecolor="white", zorder=11
    )

    (lineVx,) = axVel.plot([], [], color=NAVY, label=r"$v_X$")
    (lineVy,) = axVel.plot([], [], color=OCHRE, label=r"$v_Y$")
    (lineTheta,) = axOrient.plot([], [], color=NAVY, label=r"$\theta$")
    (lineOmega,) = axOrient.plot([], [], color=OCHRE, label=r"$\omega$")
    (lineLeftThr,) = axThr.plot([], [], color=NAVY, label=r"$T_{L}$")
    (lineRightThr,) = axThr.plot([], [], color=OCHRE, label=r"$T_{R}$")

    for ax in [axVel, axOrient, axThr]:
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()

    # ------------------------------------------------ Update Logic ------------------------------------------------
    def update(frame):
        # Determine the active trajectory
        idxTraj = np.searchsorted(fStart, frame, side="right") - 1
        frRel = frame - fStart[idxTraj]

        dataTraj = trajs[idxTraj]

        # Reset Plots at the start of a new trajectory
        if frRel == 0:
            # Re-scale axes for the new trajectory data with padding
            vx, vy = dataTraj[:, 2], dataTraj[:, 3]
            th, om = dataTraj[:, 4], dataTraj[:, 5]

            minV, maxV = min(vx.min(), vy.min()), max(vx.max(), vy.max())
            padV = (maxV - minV) * 0.1 if maxV != minV else 0.1
            axVel.set_ylim(minV - padV, maxV + padV)

            minO, maxO = min(th.min(), om.min()), max(th.max(), om.max())
            padO = (maxO - minO) * 0.1 if maxO != minO else 0.1
            axOrient.set_ylim(minO - padO, maxO + padO)

            # Reset X-axis to the length of the current trajectory
            for ax in [axVel, axOrient, axThr]:
                ax.set_xlim(0, len(dataTraj))

            # Update titles to show progress
            axTraj.set_title(f"Trajectory {idxTraj + 1}/{numTraj}")

        # Extract current state
        X, Y = dataTraj[:, 0], dataTraj[:, 1]
        vx, vy = dataTraj[:, 2], dataTraj[:, 3]
        theta, omega = dataTraj[:, 4], dataTraj[:, 5]
        lThr, rThr = dataTraj[:, 6], dataTraj[:, 7]

        # Update Artists (Standard Logic)
        if frRel >= 2:
            pts = np.array([X[:frRel], Y[:frRel]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc.set_segments(segments)
            lc.set_array(np.linspace(0, len(dataTraj), frRel - 1))
            lc.set_norm(plt.Normalize(0, len(dataTraj)))

            mask = np.arange(0, frRel, 10)
            if len(mask) > 0:
                scatter.set_offsets(np.c_[X[mask], Y[mask]])
                scatter.set_array(mask)
                scatter.set_norm(plt.Normalize(0, len(dataTraj)))

        # Drone Viz
        currTheta = theta[frRel]
        dx, dy = lARMS * math.cos(currTheta), lARMS * math.sin(currTheta)
        lx, ly = X[frRel] - dx, Y[frRel] - dy
        rx, ry = X[frRel] + dx, Y[frRel] + dy

        droneFrame.set_data([lx, rx], [ly, ry])
        lRotor.set_data([lx], [ly])
        rRotor.set_data([rx], [ry])

        lCol, rCol = _thrustColors(lThr[frRel]), _thrustColors(rThr[frRel])
        lRotor.set_markerfacecolor(lCol)
        lRotor.set_markeredgecolor(lCol)
        rRotor.set_markerfacecolor(rCol)
        rRotor.set_markeredgecolor(rCol)
        droneFrame.set_color(_thrustColors((lThr[frRel] + rThr[frRel]) / 2))

        # Parameter Lines
        tAxis = np.arange(frRel + 1)
        lineVx.set_data(tAxis, vx[: frRel + 1])
        lineVy.set_data(tAxis, vy[: frRel + 1])
        lineTheta.set_data(tAxis, theta[: frRel + 1])
        lineOmega.set_data(tAxis, omega[: frRel + 1])
        lineLeftThr.set_data(tAxis, lThr[: frRel + 1])
        lineRightThr.set_data(tAxis, rThr[: frRel + 1])

        return (
            lc,
            scatter,
            droneFrame,
            lRotor,
            rRotor,
            lineVx,
            lineVy,
            lineTheta,
            lineOmega,
            lineLeftThr,
            lineRightThr,
        )

    # ------------------------------------------------ Render ------------------------------------------------
    anim = animation.FuncAnimation(fig, update, frames=tFrames, interval=30, blit=False)

    with tqdm(
        total=tFrames, desc="Animating Trajectories", unit=" frame", ncols=250
    ) as pbar:
        anim.save(
            outFile,
            writer="ffmpeg",
            fps=30,
            progress_callback=lambda i, n: pbar.update(1),
        )

    plt.close(fig)
    print(f"\n[DONE] Unified video saved to {outFile}")


if __name__ == "__main__":
    dExpert = np.load(os.path.join("data", "expert.npz"), allow_pickle=True)
    createTrajAnim(dExpert)
