# Setpoint Shaping and Pointing Control for Satellite Repositioning

This document describes the design, analysis, and validation of this GitHub project. Since my background is in Systems and Control, I put extra emphasis on the dynamics and control aspects of the work.

I’ve written this in a conversational style on purpose. The goal isn’t to be a step by step construction manual or a formal academic paper, but to capture my thought process and explain the rationale behind key design decisions.

So if you don’t mind a casual tone and a slightly unconventional writing style, feel free to read on! 😀👍




## Table of Contents

1. [Problem Statement and Requirements](#1-problem-statement-and-requirements)
2. [Spacecraft Physical Model](#2-spacecraft-physical-model)
3. [Reaction Wheel Configuration: Why This Geometry?](#3-reaction-wheel-configuration-why-this-geometry)
4. [Flexible Mode Dynamics: The Base Excitation Problem](#4-flexible-mode-dynamics-the-base-excitation-problem)
5. [From Nonlinear Basilisk Model to Linear Control Design](#5-from-nonlinear-basilisk-model-to-linear-control-design)
6. [Why PD Control? A Transfer Function Analysis](#6-why-pd-control-a-transfer-function-analysis)
7. [Gain Selection: Deriving K and P from Requirements](#7-gain-selection-deriving-k-and-p-from-requirements)
8. [Why Feedforward? The Fundamental Limitation of Feedback](#8-why-feedforward-the-fundamental-limitation-of-feedback)
9. [Why Fourth-Order Shaping? Spectral Analysis](#9-why-fourth-order-shaping-spectral-analysis)
10. [System Integration and Trajectory Tracking](#10-system-integration-and-trajectory-tracking)
11. [Validation and Results](#11-validation-and-results)
12. [Summary of Design Decisions](#12-summary-of-design-decisions)


## 1. Problem Statement and Requirements 

### 1.1 Mission Scenario

A 750 kg spacecraft must perform a $180 \degree$ [slew manoeuvre](https://en.wikipedia.org/wiki/Slew_(spacecraft)) about the Z axis in the [body frame](https://www.sbg-systems.com/glossary/body-frame/#:~:text=The%20sensor%20coordinate%20frame%20or,base%2C%20depending%20on%20the%20application.) (i.e., a yaw rotation) within 30 seconds.

In practice, achieving such a manoeuvre within this narrow time window is in the neighbourhood of “keep dreaming,” due to limitations such as [reaction wheel](https://en.wikipedia.org/wiki/Reaction_wheel) torque constraints and disturbances from [slosh dynamics](https://en.wikipedia.org/wiki/Slosh_dynamics), to name just a few. For the purpose of this project, we deliberately disregard these limitations and focus only on the flexible dynamics of the [solar array](https://en.wikipedia.org/wiki/Solar_panels_on_spacecraft), aiming for [arcsecond](https://en.wikipedia.org/wiki/Minute_and_second_of_arc) level post manoeuvre pointing stability to enable high resolution imaging of a comet. 

### 1.2 Core Difficulty 

The spacecraft considered in this project carries flexible solar arrays, which are typically lightly damped. When the spacecraft body accelerates during the slew manoeuvre, these arrays are excited and begin to vibrate.

Because the damping is low, and because, in the space environment, free vibrations have limited ways to dissipate energy (primarily through internal/[(structural damping)](https://innovationspace.ansys.com/courses/wp-content/uploads/2020/12/2.6.2-Fundamentals-of-Damping-New-Template.pdf), the motion can persist well beyond the end of the manoeuvre. In some cases, the resulting residual vibrations can last for minutes.

This residual motion degrades pointing accuracy and can force the spacecraft to wait for the structure to settle before imaging or performing other precision tasks, effectively putting a "brake" on the mission timeline,something engineers generally try hard to avoid.

### 1.3 Requirements

| Requirement | Value | Rationale |
|-------------|-------|-----------|
| Slew angle | 180° | Mission geometry |
| Slew time | 30 s | Operational constraint |
| Post-slew settling | < 5 s to 1 arcsec | Imaging window |
| Residual vibration | < 1 mm modal displacement | Payload requirement 

 *Note:- These requirements are defined solely for this project and do not represent any known ongoing or past mission requirements.*


 <details>
<summary>Reasoning as to why this holds.</summary>

![Trust me, I'm an engineer meme](https://media3.giphy.com/media/v1.Y2lkPTZjMDliOTUyeWwzaDUzMHJ3amdvMno0bnpuZ3M3Nnp3ajB1eXBrYTV3MG55c2V5NyZlcD12MV9naWZzX3NlYXJjaCZjdD1n/LqW9dLVjQm3cs/giphy.gif)

</details>



 ## 2. Spacecraft Model

In this section, we’ll walk through how we build the spacecraft model. We’ll start with the nonlinear model used for simulation, and then move on to the linearised model we’ll use for controller design.

For the higher fidelity setup, we lean on Basilisk’s built in modules, mainly `spacecraft.Spacecraft()` to represent the rigid body hub, `reactionWheelStateEffector` to model the reaction wheels, and `linearSpringMassDamper` to capture the flexible dynamics of the solar arrays. Once those pieces are in place, we attach the state effectors (reaction wheels + solar arrays) to the central hub, and that gives us the full spacecraft model inside Basilisk.

After that, we’ll take a look at the linearised model derived from the nonlinear Basilisk model, since that’s what we’ll use for controller design. Along the way, I’ll also explain the rationale behind the modelling choices and the overall workflow.

*Note:- If curiosity gets the better of you and you decide to wander into Basilisk’s documentation rabbit hole, here are the links to the modules we’re using so you don’t have to hunt them down: [spacecraft](https://hanspeterschaub.info/basilisk/Documentation/simulation/dynamics/spacecraft/spacecraft.html), [reaction wheels](https://hanspeterschaub.info/basilisk/Documentation/simulation/dynamics/reactionWheels/reactionWheelStateEffector.html), and [linear spring-mass-damper](https://hanspeterschaub.info/basilisk/Documentation/simulation/dynamics/LinearSpringMassDamper/linearSpringMassDamper.html).* 😁

### 2.1 Hub Properties
We define our hub as :

$$
 m_{hub} = 750 kg \tag{1}
$$


$$
I_{hub} = \begin{bmatrix} 900 & 0 & 0 \\ 0 & 800 & 0 \\ 0 & 0 & 600 \end{bmatrix} \text{ kg} \cdot \text{m}^2 \tag{2}
$$

The goal here is to pick values for $m_{hub}$ and $I_{hub}$ that are representative of a medium sized satellite. I’ve chosen a *diagonal* inertia tensor on purpose; this corresponds to defining the body frame so it aligns with the hub’s principal axes of inertia, which keeps the rotational dynamics nice and simple.

### 2.2 Euler's Rotational Equations of Motion

To capture the hub’s rigid body dynamics, we use Euler’s rotational equation. In the body frame, the attitude dynamics follow from conservation of angular momentum:

$$
I_{hub}\,\dot{\boldsymbol{\omega}} \;+\; \boldsymbol{\omega} \times \left(I_{hub}\,\boldsymbol{\omega}\right)
\;=\; \boldsymbol{\tau}_{ext} \tag{3}
$$

Here, $I_{hub}$ is the inertia tensor defined above, and $\omega $ and $ \dot\omega$  are the hub’s angular velocity and angular acceleration, respectively.

Expanding the equation into components (for our diagonal inertia tensor):

$$
I_{xx}\dot{\omega}_x \;-\; (I_{yy}-I_{zz})\,\omega_y\omega_z \;=\; \tau_x \tag{4}
$$

$$
I_{yy}\dot{\omega}_y \;-\; (I_{zz}-I_{xx})\,\omega_z\omega_x \;=\; \tau_y \tag{5}
$$

$$
I_{zz}\dot{\omega}_z \;-\; (I_{xx}-I_{yy})\,\omega_x\omega_y \;=\; \tau_z \tag{6}
$$

The terms $(I_{yy}-I_{zz})$,$\omega_y\omega_z$, etc., are the **gyroscopic coupling terms**. They transfer angular momentum between axes during multi axis rotation.

*Note: I found an interesting reference to angular momentum transfer, and its removal from a spacecraft using gravity gradient moments. Even though its from the late 60's i still found it quiet inetersting. You can find the documentation [here](https://ntrs.nasa.gov/api/citations/19660011641/downloads/19660011641.pdf).*


### 2.3 Modified Rodrigues Parameters (MRPs)

To parameterize our attitude representation, we’ll use [MRPs](https://ntrs.nasa.gov/api/citations/19960035754/downloads/19960035754.pdf) instead of [quaternions](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation) or Euler angles. The reason I’m going with MRPs comes down to three main perks:

- Minimal representation : (only 3 parameters)

- No [gimbal lock](https://en.wikipedia.org/wiki/Gimbal_lock) unlike Euler angles 

- Quadratic kinematics : the kinematic equation is a polynomial and not [transcendental](https://en.wikipedia.org/wiki/Transcendental_equation)

So, for a rotation of angle $ \Phi$ about an axis $ \hat e$ , the MRP can be computed as :

$$\boldsymbol{\sigma} = \hat{\mathbf{e}} \tan\left(\frac{\Phi}{4}\right) \tag{7}$$

*Note:- notice how MRPs are computed? we can leverage this formula to force a small angle approximation to linearise our plant to design our control system down the lane.*

With the MRP computation now defined, we’ll leverage the MRP representation and define the [kinematic differential equation](https://digitalcommons.odu.edu/cgi/viewcontent.cgi?article=1374&context=mae_etds)  as:

$$

\dot\sigma = \frac{1}{4} B(\sigma)\omega \tag{8}

$$

Where:

$$

B(\sigma)=(1-\sigma^2)I_3 + 2[\sigma_\times]+2\sigma\sigma^T \tag{9}

$$

and as we surely recall from vector algebra lectures that:

$$
 \sigma^2 = \sigma^T\sigma \tag{10}
 
$$


## 3. Reaction Wheels




