# Setpoint Shaping and Pointing Control for Satellite Repositioning

This document describes the design, analysis, and validation of this GitHub project. Since my background is in Systems and Control, I put extra emphasis on the dynamics and control aspects of the work.

I’ve written this in a conversational style on purpose. The goal isn’t to be a step by step construction manual or a formal academic paper, but to capture my thought process and explain the rationale behind key design decisions.

So if you don’t mind a casual tone and a slightly unconventional writing style, feel free to read on!




## Table of Contents

1. [Problem Statement and Requirements](#1-problem-statement-and-requirements)
2. [Spacecraft Model](#2-spacecraft-model)
3. [Reaction Wheels ](#3-reaction-wheels)
4. [Flexible Mode Dynamics](#4-flexible-mode-dynamics)
5. [Linearization of the Spacecraft Model](#5-linearization-of-the-spacecraft-model)
6. [Feedback Control Design](#6-feedback-control-design)
7. [Feedforward Control Design](#7-feedforward-control-design)
8. [Verification and Validation](#8-verification-and-validation)
9. [Conclusion](#9-conclusion)



## 1. Problem Statement and Requirements 

### 1.1 Mission Scenario

A 750 kg spacecraft must perform a $`180 \degree`$ [slew manoeuvre](https://en.wikipedia.org/wiki/Slew_(spacecraft)) about the Z axis in the [body frame](https://www.sbg-systems.com/glossary/body-frame/#:~:text=The%20sensor%20coordinate%20frame%20or,base%2C%20depending%20on%20the%20application.) (i.e., a yaw rotation) within 30 seconds.

In practice, achieving such a manoeuvre within this narrow time window is in the neighbourhood of “keep dreaming,” due to limitations such as [reaction wheel](https://en.wikipedia.org/wiki/Reaction_wheel) torque constraints and disturbances from [slosh dynamics](https://en.wikipedia.org/wiki/Slosh_dynamics), to name just a few. For the purpose of this hobby project, we deliberately disregard these limitations and focus only on the flexible dynamics of the [solar array](https://en.wikipedia.org/wiki/Solar_panels_on_spacecraft), aiming for [arcseconds](https://en.wikipedia.org/wiki/Minute_and_second_of_arc) level post manoeuvre pointing stability to enable high resolution imaging of a comet. 

### 1.2 Core Difficulty 

The spacecraft considered in this project carries flexible solar arrays, which are typically lightly damped. When the spacecraft body accelerates during the slew manoeuvre, these arrays are excited and begin to vibrate.

Because the damping is low, and because, in the space environment, free vibrations have limited ways to dissipate energy (primarily through internal/[(structural damping)](https://innovationspace.ansys.com/courses/wp-content/uploads/2020/12/2.6.2-Fundamentals-of-Damping-New-Template.pdf), the motion can persist well beyond the end of the manoeuvre. In some cases, the resulting residual vibrations can last for minutes.

This residual motion degrades pointing accuracy and can force the spacecraft to wait for the structure to settle before imaging or performing other precision tasks, effectively putting a "brake" on the mission timeline,something engineers generally try hard to avoid.

### 1.3 Requirements

<div align="center">

| Requirement | Value | Rationale |
|-------------|-------|-----------|
| Slew angle | 180° | Comet observation (coma) |
| Slew time | 30 s | Mission requirement |
| Post slew settling |  RMS ≤7 arcsec (within 60s) | Imaging requirement |
| Post slew array acceleration | RMS < 10 $`mm/s^2`$ modal acceleration | Safety requirement 
|Phase margin | 70°-75°| Robustness requirement

</div>

 *Note:- These requirements are defined solely for this project and do not represent any known ongoing or past mission requirements.*

 ## 2. Spacecraft Model

In this section, we’ll walk through how we build the spacecraft model. We’ll start with the nonlinear model used for simulation, and then move on to the linearised model we’ll use for controller design.

For the higher fidelity setup, we lean on Basilisk’s built in modules, mainly `spacecraft.Spacecraft()` to represent the rigid body hub, `reactionWheelStateEffector` to model the reaction wheels, and `linearSpringMassDamper` to capture the flexible dynamics of the solar arrays. Once those pieces are in place, we attach the state effectors (reaction wheels + solar arrays) to the central hub, and that gives us the full spacecraft model inside Basilisk.

After that, we’ll take a look at the linearised model derived from the nonlinear Basilisk model, since that’s what we’ll use for controller design. Along the way, I’ll also explain the rationale behind the modelling choices and the overall workflow.

*Note:- If curiosity gets the better of you and you decide to wander into Basilisk’s documentation rabbit hole, here are the links to the modules we’re using so you don’t have to hunt them down: [spacecraft](https://hanspeterschaub.info/basilisk/Documentation/simulation/dynamics/spacecraft/spacecraft.html), [reaction wheels](https://hanspeterschaub.info/basilisk/Documentation/simulation/dynamics/reactionWheels/reactionWheelStateEffector.html), and [linear spring-mass-damper](https://hanspeterschaub.info/basilisk/Documentation/simulation/dynamics/LinearSpringMassDamper/linearSpringMassDamper.html).* 

### 2.1 Hub Properties
We define our hub as :

```math
\begin{aligned}
m_{hub} = 750 kg 
\end{aligned}
```


```math
\begin{aligned}
I_{hub} = \begin{bmatrix} 900 & 0 & 0 \\ 0 & 800 & 0 \\ 0 & 0 & 600 \end{bmatrix} \text{ kg} \cdot \text{m}^2
\end{aligned}
```

The goal here is to pick values for $`m_{hub}`$ and $`I_{hub}`$ that are representative of a medium sized satellite. I’ve chosen a *diagonal* inertia tensor on purpose; this corresponds to defining the body frame so it aligns with the hub’s principal axes of inertia, which keeps the rotational dynamics nice and simple.

### 2.2 Euler's Rotational Equations of Motion

To capture the hub’s rigid body dynamics, we use Euler’s rotational equation. In the body frame, the attitude dynamics follow from conservation of angular momentum:

```math
\begin{aligned}
I_{hub}\dot{\boldsymbol{\omega}}+\boldsymbol{\omega} \times \left(I_{hub}\boldsymbol{\omega}\right)=\boldsymbol{\tau}_{ext}
\end{aligned}
```

Here, $`I_{hub}`$ is the inertia tensor defined above, and $`\omega`$ and $`\dot\omega`$ are the hub's angular velocity and angular acceleration, respectively.

Expanding the equation into components (for our diagonal inertia tensor):

```math
\begin{aligned}
I_{xx}\dot{\omega}_x - (I_{yy}-I_{zz})\omega_y\omega_z = \tau_x
\end{aligned}
```

```math
\begin{aligned}
I_{yy}\dot{\omega}_y - (I_{zz}-I_{xx})\omega_z\omega_x = \tau_y
\end{aligned}
```

```math
\begin{aligned}
I_{zz}\dot{\omega}_z - (I_{xx}-I_{yy})\omega_x\omega_y = \tau_z
\end{aligned}
```

The terms $`(I_{yy}-I_{zz})\omega_y\omega_z`$ etc., are the gyroscopic coupling terms. They transfer angular momentum between axes during multi axis rotation.

*Note: I found an interesting reference to angular momentum transfer, and its removal from a spacecraft using gravity gradient moments. Even though its from the late 60's i still found it quiet inetersting. You can find the documentation [here](https://ntrs.nasa.gov/api/citations/19660011641/downloads/19660011641.pdf).*


### 2.3 Modified Rodrigues Parameters (MRPs)

To parameterize our attitude representation, we’ll use [MRPs](https://ntrs.nasa.gov/api/citations/19960035754/downloads/19960035754.pdf) instead of [quaternions](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation) or Euler angles. The reason I’m going with MRPs comes down to three main perks:

- Minimal representation : (only 3 parameters)

- No [gimbal lock](https://en.wikipedia.org/wiki/Gimbal_lock) unlike Euler angles 

- Quadratic kinematics : the kinematic equation is a polynomial and not [transcendental](https://en.wikipedia.org/wiki/Transcendental_equation)

So, for a rotation of angle $`\Phi`$ about an axis $`\hat e`$, the MRP can be computed as :

```math
\begin{aligned}
\boldsymbol{\sigma} = \hat{\mathbf{e}} \tan\left(\frac{\Phi}{4}\right)
\end{aligned}
```

*Note:- notice how MRPs are computed? we can leverage this formula to force a small angle approximation to linearise our plant to design our control system down the lane.*

With the MRP computation now defined, we’ll leverage the MRP representation and define the [kinematic differential equation](https://digitalcommons.odu.edu/cgi/viewcontent.cgi?article=1374&context=mae_etds)  as:

```math
\begin{aligned}
\dot\sigma = \frac{1}{4} B(\sigma)\omega
\end{aligned}
```

Where:

```math
\begin{aligned}
B(\sigma)=(1-\sigma^2)I_3 + 2[\sigma_\times]+2\sigma\sigma^T
\end{aligned}
```

and as we surely recall from vector algebra lectures that:

```math
\begin{aligned}
\sigma^2 = \sigma^T\sigma
\end{aligned}
```


## 3. Reaction Wheels



Alright , now that the hub model is out of the way, we can start building our first `StateEffector`. This one is also the main (and basically *only*) actuator we’ll use in this project; the reaction wheels.

Reaction wheels let the spacecraft rotate by exploiting conservation of angular momentum. In simple terms, spinning a wheel inside the spacecraft causes the spacecraft body to rotate in the opposite direction about the same axis!. If you want a quick visual for what’s going on, i feel this [YouTube Short](https://www.youtube.com/shorts/oaGZo1ZVw7g) explains it nicely.

### 3.1 Torque Source

To characterize the reaction wheels and how they influence the spacecraft’s rotation, we model them as a torque source. That said, we *don’t* treat them as an ideal torque actuator as we explicitly account for the wheel/rotor inertia.

With that in mind, the torque produced by the \(i\) th reaction wheel can be written as:

```math
\begin{aligned}
\tau_{RW,i} = -J_{s,i} \dot\Omega_i \hat g_{s,i}
\end{aligned}
```

Here, $`J_s`$ is the rotor inertia and $`\Omega_i`$ is the wheel spin rate, so the inertia matrix becomes:

```math
\begin{aligned}
J_s = \begin{bmatrix} 0.05 & 0 & 0 \\ 0 & 0.05 & 0 \\ 0 & 0 & 0.05 \end{bmatrix} kg \cdot m^2
\end{aligned}
```

Using this, the total torque from the full reaction wheel set can be expressed as:

```math
\begin{aligned}
\tau_{RW} = -G_s J_s \dot\Omega
\end{aligned}
```

This can be rewritten more compactly as:

```math
\begin{aligned}
\tau_{RW} = -G_su_{motor}
\end{aligned}
```

where $`u_{motor} = J_s\dot\Omega`$ is the vector of motor torques.

Since $`G_s`$ is the spin axis matrix defined by:

```math
\begin{aligned}
\mathbf{G}_s = \begin{bmatrix} | & | & | \\ \hat{\mathbf{g}}_{s1} & \hat{\mathbf{g}}_{s2} & \hat{\mathbf{g}}_{s3} \\ | & | & | \end{bmatrix}
\end{aligned}
```

we can see from the compact torque expression that the torque we get depends directly on how we choose the wheel spin axes. So before we go any further, we need to pick those axes 
which brings us to our first real design decision.

### 3.2 Actuator Alignment 

To figure out the best way to orient our reaction wheels, we first need to understand how the wheel geometry affects both the spacecraft’s control authority and the numerical behaviour of our allocation. And honestly, what better way to build that intuition than running a parameter sweep, right?

Before we run the sweep, I’m going to explicitly align one wheel with the body Z axis. The main reason is simple; our big repositioning manoeuvre is about Z (yaw), so having one wheel dedicated to the Z axis gives us clean, direct authority on the exact axis we care about most. That leaves the orientation of the other two wheels as the real design choice.

Lets properly write out our spin axis matrix $`G_s`$ from the definition above, so we get:

```math
\begin{aligned}
G_s = \begin{bmatrix} + sin(\alpha) & - sin(\alpha) & 0 \\ + cos(\alpha) & + cos(\alpha) & 0 \\ 0 & 0 & 1 \end{bmatrix}
\end{aligned}
```

From this matrix, we can see that the wheel aligned with the body Z axis has full control authority in Z, and contributes no torque about the body X and Y axes. For the other two wheels, we deliberately share control authority over the XY plane, so we don’t end up relying on a single wheel for a single axis (which helps reduce individual wheel effort by spreading the load).

Notice how the cosine terms are both positive; this is intentional. The X axis authority comes from the difference in wheel torques (because the X components are opposite), while the Y axis authority comes from the sum (because the Y components are the same and positive).

Suppose you had flipped one cosine term to negative, you’d effectively point one wheel toward \(-Y\), and the Y components would start cancelling out. That would reduce your Y authority, which is not what we want.

Now we pick our rotation axes angles $`\alpha`$, based on the following paramteres:

- [Condition number](https://en.wikipedia.org/wiki/Condition_number) of the allocation matrix $`G_sG_s^T`$ : We aim for a lower condition number as it represents better numerical conditioning and more isotropic control.

- Axis wise torque autority: we compute the max body torque achievable along X and Y for a fixed wheel torque limit, since the control authority changes with the angle $`\alpha`$


With our parameter sweep now complete, we analyse the results given in the figures below:




<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="plots\image.png" width="300">
  <img src="plots\image-1.png" width="300">
</div>


After analysing the plots, we quickly notice that the a cant angle ($`\alpha`$) of $`45\degree`$ has the lowest conditioning number. Furthermore, we also notice that the same $`\alpha`$ value gives equal torque authority along both the X and Y axes (for the same wheel torque limit).

*Note:- Equal X/Y authority gives a more uniform control capability over the XY plane, which makes [gain balancing](https://patentscope.wipo.int/search/en/WO2024245715) a lot less painful (the effort is naturally shared).*

Thus from these analyses, we decide our cant angle to be set to $`\alpha = 45 \degree`$

Hence the spin axis matrix $`G_s`$ becomes:

```math
\begin{aligned}
\mathbf{G}_s = \begin{bmatrix} \frac{\sqrt{2}}{2} & -\frac{\sqrt{2}}{2} & 0 \\ \frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2} & 0 \\ 0 & 0 & 1 \end{bmatrix}
\end{aligned}
```

With the spin axis determined, we ask ourselves " can this orientation enable control on all three axises?" or more mathematically, " can the torque mapping span $`\mathbb{R}^3`$?" we answer this by looking at the rank of this matrix.

For full 3 axis attitude control, the reaction wheel spin axes must span $`\mathbb{R}^3`$. Mathematically, the spin axis matrix from the definition:

```math
\begin{aligned}
\mathbf{G}_s = \begin{bmatrix} | & | & | \\ \hat{\mathbf{g}}_{s1} & \hat{\mathbf{g}}_{s2} & \hat{\mathbf{g}}_{s3} \\ | & | & | \end{bmatrix}
\end{aligned}
```

must have rank 3 (be invertible for a 3 wheel system).

Let's verify our configuration:

```math
\begin{aligned}
\mathbf{G}_s = \begin{bmatrix} \frac{\sqrt{2}}{2} & -\frac{\sqrt{2}}{2} & 0 \\ \frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2} & 0 \\ 0 & 0 & 1 \end{bmatrix}
\end{aligned}
```

Computing the determinant:

```math
\begin{aligned}
\det(\mathbf{G}_s) = 1 \cdot \det\begin{bmatrix} \frac{\sqrt{2}}{2} & -\frac{\sqrt{2}}{2} \\ \frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2} \end{bmatrix} = 1 \cdot \left(\frac{1}{2} + \frac{1}{2}\right) = 1
\end{aligned}
```

Since $`\det(\mathbf{G}_s) = 1 \neq 0`$, the system has full 3 axis controllability.

*Note :- In many spacecraft, reaction wheels are mounted rigidly (body fixed). Systems that use gimbals to reorient a spinning actuator are more commonly referred to as control moment gyros ([CMGs](https://en.wikipedia.org/wiki/Control_moment_gyroscope)), or gimballed wheel setups. In this simulation, the reaction wheels are assumed to be fixed at a canted orientation, so there is no gimbal and no controllability over the wheel spin axis direction. therefore only the wheel torque is commanded.*

## 4 Flexible Mode Dynamics

With the reaction wheels now set up and the cant angle locked in, we can move on to our second `StateEffector`; the flexible solar arrays. As shown in the figure below, the arrays are mounted along the body Y axis.

Now, the sketch isn’t exactly a work of art , but it does the job! it shows where the [modal mass](https://www.sciencedirect.com/topics/engineering/modal-mass) is located and how far it sits from the spacecraft hub. In this section, we’ll dig into why the arrays get excited (and start vibrating) during a slew, we’ll try (our best!) to figure out which axis the dominant vibration shows up on, and finally we’ll look at how the `LinearSpringMassDamper` module models the array dynamics.

<div style="display: flex; justify-content: center; gap: 10px;">

<image src = "plots\image-2.png" width = 600>

</div>

### 4.1 Effects of Slew Manoeuvre on Array Vibration

To get an intuition for what the slew does to the arrays, let’s momentarily pretend the hub is a perfectly rigid rotating body. When the hub rotates, any point attached to it experiences inertial (absolute) acceleration.

Consider a point on the array located at

```math
\begin{aligned}
r = \begin{bmatrix} 0 & r_y & 0 \end{bmatrix}^T
\end{aligned}
```

i.e., it sits a distance $`r_y`$ away from the hub center along the body \(Y\) axis.

The absolute acceleration of this point is:

```math
\begin{aligned}
\ddot r_{abs} = \dot \omega \times r + \omega \times (\omega \times r )
\end{aligned}
```

For a pure yaw rotation $`\omega = [0,0,\omega_z]^T`$ (which is exactly the kind of motion we have during the slew), the tangential term becomes:

```math
\begin{aligned}
\dot \omega \times r = \begin{bmatrix} 0 \\ 0 \\ \dot \omega_z \end{bmatrix} \times \begin{bmatrix} 0 \\ r_y \\ 0 \end{bmatrix} =  \begin{bmatrix} -\dot \omega_z r_y \\ 0 \\ 0 \end{bmatrix}
\end{aligned}
```

And the centripetal acceleration is:

```math
\begin{aligned}
\boldsymbol{\omega} \times (\boldsymbol{\omega} \times \mathbf{r}) = \begin{bmatrix} 0 \\ -\omega_z^2 r_y \\ 0 \end{bmatrix}
\end{aligned}
```

Notice what the tangential term is telling us? the tangential acceleration points along the body \(X\) axis (perpendicular to both the rotation axis \(Z\) and the array direction \(Y\)). Intuitively, when the hub accelerates in yaw, it “drags” the array attachment points sideways in the tangential direction. The array tip has inertia, so it can’t follow instantly and it lags behind, the array bends, and that’s how the vibration starts.

*Note :- By looking at the phenomenon above, we can  frame it as a [base excitation problem](https://community.wvu.edu/~bpbettig/MAE340/Lecture_2_4_Base_excitation.pdf). This is a classic concept taught in the " Advanced Vibration " course modules.*

### 4.2 Modal Equation of Motion

In the previous subsection we saw why the slew excites the solar array. Now let’s talk about how we can characterise that vibration.

A super common way to do this is to approximate the flexible motion with a [mass spring damper model](https://en.wikipedia.org/wiki/Mass-spring-damper_model). It’s not trying to capture every tiny structural detail, but it does a great job at capturing the stuff we actually care about! the vibration frequency, the amplitude of the response (for a given excitation), and how fast the oscillations die out thanks to damping.

For a modal coordinate $`\rho`$, we form our equation of motion as:

```math
\begin{aligned}
m\ddot\rho + c \dot \rho + k \rho = F_{excitation}
\end{aligned}
```

where:
- $`\rho`$ is the modal displacement (bending deflection)
- $`m`$ is the effective modal mass
- $`c = 2\zeta\omega_n m`$ is the damping coefficient
- $`k = \omega_n^2 m`$ is the spring stiffness
- $`F_{excitation}`$ is the basen excitation (inertial) forcing term

The excitation comes from the tangential base acceleration during the yaw slew. Using $`a_{tangential} = -\dot{\omega}_z r_y`$ from the tangential acceleration expression, the equivalent inertial forcing can be written as:

```math
\begin{aligned}
F_{excitation} = -m \cdot a_{tangential} = -m \cdot (-\dot{\omega}_z r_y) = m \cdot r_y \cdot \dot{\omega}_z
\end{aligned}
```

for yaw manoeuvre (so we can neglect the cross coupling term from the Z-axis dynamics), we use:

```math
\begin{aligned}
\dot{\omega}_z = \tau_z / I_{zz}
\end{aligned}
```

and substitute into the modal equation:

```math
\begin{aligned}
m\ddot{\rho} + c\dot{\rho} + k\rho = \frac{m \cdot r_y}{I_{zz}} \cdot \tau_z
\end{aligned}
```

Dividing by $`m`$ gives the standard second order form:

```math
\begin{aligned}
\ddot{\rho} + 2\zeta\omega_n\dot{\rho} + \omega_n^2\rho = \frac{r_y}{I_{zz}} \cdot \tau_z
\end{aligned}
```

This reveals the modal gain:

```math
\begin{aligned}
G_{modal} = \frac{r_y}{I_{zz}}
\end{aligned}
```

### 4.4 Modal Parameters

<div align="center">

| Mode | Frequency $`f_n`$ | $`\omega_n`$ (rad/s) | Damping $`\zeta`$ | Lever Arm $`r`$ | Modal Gain |
|------|-----------------|-------------------|-----------------|---------------|------------|
| 1 | 0.4 Hz | 2.51 | 0.02 | 3.5 m | 0.0056 |
| 2 | 1.3 Hz | 8.17 | 0.015 | 4.5 m | 0.0073 |

</div>


*Note :- i picked the modal parameters to roughly represent a large solar array. In general, big arrays tend to have very low natural frequencies. Also, pure material (structural) damping is usually quite small, but in real hardware the attachment points, fasteners, and joints add a bit of extra damping. Taking all that into account, a damping ratio in the ballpark of 0.01 to 0.02 is what I kept seeing during my literature review. If you think I’ve overestimated the damping, please reach out! I’m happy to revisit it and update both this documentation and the codes 👍*

### 4.5 Settling Time for A Second Order Oscillator

Since we’ve modelled the vibration as a second order mass spring damper system, we can estimate its settling time using a standard percent criterion. The most common choices are the 2% or 5% settling time definitions. For this project, I’ll use the 2% criterion, which gives the well known approximation:

```math
\begin{aligned}
t_s \approx \frac{4}{\zeta \omega_n}
\end{aligned}
```

For Mode 1 ($`\zeta = 0.02`$, $`\omega_n = 2.51`$ rad/s):

```math
\begin{aligned}
t_s \approx \frac{4}{0.02 \times 2.51} = \frac{4}{0.05} = 80 \text{ seconds}
\end{aligned}
```

So, this mode takes on the order of 80 seconds to settle, which is more than twice our slew time! That’s exactly why we’d like to avoid (as much as possible) exciting these flexible mode frequencies during the manoeuvre.

*Note :- if you are wondering about the second mode, the settling time is around 24.48 seconds!*

## 5 Linearization of The Spacecraft Model

The Basilisk simulation is running the full nonlinear model, which includes:

1. **Nonlinear kinematics**: $`\dot{\boldsymbol{\sigma}} = \frac{1}{4}\mathbf{B}(\boldsymbol{\sigma})\boldsymbol{\omega}`$
2. **Gyroscopic coupling**: $`\boldsymbol{\omega} \times \mathbf{I}\boldsymbol{\omega}`$
3. **Reaction wheel momentum**: the wheel angular momentum contributes to (and couples into) the total spacecraft momentum
4. **Flexible mode coupling**: the flexible modes exchange momentum with the hub

Which naturally leads to the next question, under what conditions can a linearised version of this model still do a decent job of matching the real behaviour?

My take is pretty straightforward here. linearisation works best when the nonlinear terms either vanish entirely or become small enough to ignore. So the game plan is to figure out when that happens, and then design our controller around that linearised plant.

### 5.1 Principal Axis Rotation and Gyroscopic Coupling

Since our slew manoeuvre is about the body Z axis, it's worth checking what that does to the gyroscopic coupling terms. To do that, let's look at the Z-axis component from Euler's equations:

```math
\begin{aligned}
I_{zz}\dot{\omega}_z - (I_{xx} - I_{yy})\omega_x\omega_y = \tau_z
\end{aligned}
```

Now, if we enforce a pure yaw motion by setting $`\omega_x \omega_y = 0`$ (i.e., no roll/pitch rotation), the coupling term drops out and we're left with:

```math
\begin{aligned}
I_{zz} \dot \omega_z = \tau_z
\end{aligned}
```

And importantly, this isn’t an approximation, it’s exact! So for a yaw only manoeuvre (assuming we don’t pick up roll/pitch motion from disturbances or coupling), the Z axis dynamics are perfectly linear!

### 5.2 Small Angle Approximation of MRP Kinematics

To see if we can perform a small angle approx of MRP kinematics, we refer to the kinematic differential equation and the B matrix definition, and we write:

```math
\begin{aligned}
\dot{\boldsymbol{\sigma}} = \frac{1}{4}\left[(1 - \sigma^2)\mathbf{I}_3 + 2[\boldsymbol{\sigma}\times] + 2\boldsymbol{\sigma}\boldsymbol{\sigma}^T\right]\boldsymbol{\omega}
\end{aligned}
```

For small $`|\boldsymbol{\sigma}|`$ (small attitude errors):

```math
\begin{aligned}
\dot{\boldsymbol{\sigma}} \approx \frac{1}{4}\boldsymbol{\omega}
\end{aligned}
```

So this has linearised our kinematics!

*Note :- This actually highlights an important constraint for our controller design. the small angle (linearised) plant only behaves linear when the attitude errors stay small. So we can’t expect a small angle linear feedback controller to take us from a huge initial error (like a 180° slew) all the way to the target by itself! that’s way outside the region where the linear model is valid.*

*Instead, we use a tracking setup. a feedforward generates an instantaneous reference trajectory (and the corresponding torque/motion profile), and the feedback controller’s job is mainly to correct small deviations around that trajectory. In other words, the feedforward does the heavy lifting, and the feedback keeps us on the rails!*

### 5.3 Linearized Model

Combining the above:

**Dynamics:**
```math
\begin{aligned}
I_{zz}\dot{\omega}_z = \tau_z
\end{aligned}
```

**Kinematics (single axis):**
```math
\begin{aligned}
\dot{\sigma}_z = \frac{1}{4}\omega_z
\end{aligned}
```

Taking Laplace transforms:

```math
\begin{aligned}
sI_{zz}\Omega(s) = T(s)
\end{aligned}
```

```math
\begin{aligned}
s\Sigma(s) = \frac{1}{4}\Omega(s)
\end{aligned}
```

Eliminating $`\Omega(s)`$:

```math
\begin{aligned}
\Sigma(s) = \frac{1}{4s}\Omega(s) = \frac{1}{4s} \cdot \frac{T(s)}{sI_{zz}}
\end{aligned}
```

```math
\begin{aligned}
G_{rigid}(s) = \frac{\Sigma(s)}{T(s)} = \frac{1}{4I_{zz}s^2}
\end{aligned}
```

This is basically a double [integrator](https://electronics.stackexchange.com/questions/333888/why-is-the-frequency-domain-representation-of-an-integrator-1-s)! two poles sitting at the origin. In frequency domain terms, it carries about $`180 \degree`$ of phase lag right out of the gate, which is not unstable by itself (it’s marginal in open loop), but it definitely makes the feedback design a tad bit more challenging.

*Note :- As you’ve probably already guessed, I’m a little biased toward frequency domain control (life is just easier here)*

### 5.4 Flex Modes to Plant 

For each flexible mode, the transfer function from torque to modal displacement is:

```math
\begin{aligned}
\frac{P_i(s)}{T(s)} = \frac{G_{modal,i}}{s^2 + 2\zeta_i\omega_{n,i}s + \omega_{n,i}^2}
\end{aligned}
```

The complete plant from torque to attitude includes the coupling:

```math
\begin{aligned}
G_{flex}(s) = \frac{1}{4I_{eff}s^2} \cdot \prod_{i=1}^{N_{modes}} \frac{s^2 + 2\zeta_i\omega_i s + \omega_i^2 + \Delta_i}{s^2 + 2\zeta_i\omega_i s + \omega_i^2}
\end{aligned}
```

where $`I_{eff}`$ includes the effective inertia from modal masses, and $`\Delta_i`$ represents the anti resonance shifts from coupling.

So yeah, in the Bode plot of our flexible plant (torque $`\rightarrow`$ attitude) below, you can clearly spot both the resonance and the anti resonance.


<div style="display: flex; justify-content: center; gap: 10px;">

<image src = "plots\image-3.png" width = 500>



</div>

*Note :- notice how the anti resonance shows up before the resonance? this is because our plant model is [collocated](https://www.pml.uliege.be/wp-content/uploads/2022/07/1-s2.0-S0888327022006082-main.pdf). So, each flexible mode introduces a zero (anti resoannce) slightly below its pole (resonance)*

## 6 Feedback Control Design

Before we pick a controller, let’s first get a feel for the basic properties of the plant:

<div align="center">

| Property | Value |
|----------|-------|
| Magnitude at DC | $`\infty`$ (two poles at the origin) |
| Magnitude slope | -40 dB/decade (away from flexible modes) |
| Phase | $`\approx -180°`$ (away from resonance/anti resonance) |
| Gain margin | Not very informative in the usual sense (the phase sits near $`-180°`$ for most frequencies, and flexible mode distort the standard crossover picture) |

</div>

Now, if we drive this plant with a sinusoid at a frequency that's not near a flexible mode, the output will be roughly shifted by $`-180\degree`$. In plain english, the plant mostly behaves like a sign inversion, except around the resonance/anti resonance neighbourhood where the flexible dynamics take over.

### 6.1 Nyquist Stability Criteria

Now that we have a decent feel for the plant and its key quirks, we can look at how to stabilise this (double integrator) system. More precisely, we want to choose a controller such that the open loop transfer function

```math
\begin{aligned}
L(s) = G_{flex}(s) C(s)
\end{aligned}
```

does not encircle the point $`(-1,0)`$ in the complex plane.

Away from the flexible modes, the double integrator part of the plant contributes roughly $`-180°`$ of phase. So at the [gain crossover frequency](https://www.mathworks.com/help/control/ref/dynamicsystem.margin.html) $`\omega_c`$ (where $`|L(j\omega_c)| = 1`$), we can write:

```math
\begin{aligned}
\angle L(j\omega_c) = \angle C(j\omega_c) - 180°
\end{aligned}
```

For a stable closed loop system we want a positive [phase margin](https://en.wikipedia.org/wiki/Phase_margin), i.e.:

```math
\begin{aligned}
PM = 180° + \angle L(j\omega_c) > 0
\end{aligned}
```

which (under the double integrator phase) boils down to:

```math
\begin{aligned}
\angle C(j\omega_c) > 0
\end{aligned}
```

This means that we need a controller that provides a phase lead! and points us in the direction of using a derivative term (natural phase lead!)


### 6.2 Proportional Derivative Control (PD)

Since we’ve established we need some phase lead from the controller to improve closed loop stability, the most straightforward move is to add a derivative term (because it contributes positive phase).

A PD controller can be written as:

```math
\begin{aligned}
C_{PD}(s) = K_p + K_d s
\end{aligned}
```
with $`K_p`$ and $`K_d`$ being the proportional and derivative gain respectively.

Since $`s = j\omega`$, we can evaluate our controller on the imaginary axis, yeilding:

```math
\begin{aligned}
\angle C_{PD}(j\omega) = \arctan \left(\frac{K_d\omega}{K_p}\right) \in [0°, 90°)
\end{aligned}
```
Notice that this phase contribution is always positive? The derivative term naturally provides the phase lead we need. At the crossover frequency $`\omega_c`$:

```math
\begin{aligned}
\angle L(j\omega_c) = \arctan\left(\frac{K_d\omega_c}{K_p}\right) - 180°
\end{aligned}
```
So our phase margin becomes:

```math
\begin{aligned}
PM = \arctan\left(\frac{K_d\omega_c}{K_p}\right)
\end{aligned}
```

*Note :- For example, if $`K_d\omega_c/K_p = 1`$, we get $`PM = 45°`$. If $`K_d\omega_c/K_p = 2`$, we get $`PM = 63°`$. Pretty neat, right?*

Before we move on to gain selection, it's worth pausing to think about what PD control actually does physically. In the time domain, our control law is:

```math
\begin{aligned}
\tau = -K_p\sigma_e - K_d\omega_e
\end{aligned}
```

Where $`\sigma_e`$ is the attitude error (in MRPs) and $`\omega_e`$ is the rate error.

- Proportional term ($`-K_p\sigma_e`$): This creates a torque proportional to how far off we are from the target. It's essentially a virtual spring that pulls the spacecraft toward the desired attitude.

- Derivative term ($`-K_d\omega_e`$): This creates a torque proportional to how fast we're moving relative to the target. It's essentially a virtual damper that resists motion and prevents overshoot.

*Note:-  remember how a torsional spring ends up with a static twist $`\delta`$ under a constant (DC) torque? That’s pretty similar to what happens with proportional control, you end up with a steady state error!*

So PD control is  adds virtual spring and damper dynamics to our spacecraft!

### 6.3 Controller Gain Selection

With our controller now defined, we need to perform the gain selection. though I admit, I did plenty of trial and error during development, there is actually a much more principled way to derive it from our closed loop system requirements!

Our control law in the MRP/rate domain, mentioned in the previous sectionas :

```math
\begin{aligned}
\tau = -K_p\sigma_e - K_d\omega_e
\end{aligned}
```
Since $`\dot{\sigma} \approx \omega/4`$ for small angles (remember that linearisation from earlier?), we have $`\omega \approx 4\dot{\sigma}`$. So the controller in the Laplace domain, operating on $`\sigma`$, becomes:

```math
\begin{aligned}
T(s) = -(K_p + 4K_d s)\Sigma_e(s)
\end{aligned}
```
So effectively, our controller is $`C(s) = K_p + 4K_d s`$.


With our plant $`G(s) = 1/(4I_{zz}s^2)`$, the loop transfer function becomes:

```math
\begin{aligned}
L(s) = G(s)C(s) = \frac{K_p + 4K_d s}{4I_{zz}s^2}
\end{aligned}
```

Similarly, our closed loop transfer function takes the form:

```math
\begin{aligned}
G_{CL}(s) = \frac{L(s)}{1 + L(s)} = \frac{K_p + 4K_d s}{4I_{zz}s^2 + 4K_d s + K_p}
\end{aligned}
```

Thus he [characteristic polynomial](https://en.wikipedia.org/wiki/Closed-loop_pole) of our closed loop system is:

```math
\begin{aligned}
4I_{zz}s^2 + 4K_d s + K_p = 0
\end{aligned}
```
Dividing through by $`4I_{zz}`$ we get:

```math
\begin{aligned}
s^2 + \frac{K_d}{I_{zz}}s + \frac{K_p}{4I_{zz}} = 0
\end{aligned}
```
Now, let's recall the [standard second order](https://apmonitor.com/pdc/index.php/Main/SecondOrderSystems) form:

```math
\begin{aligned}
s^2 + 2\zeta_{CL}\omega_n s + \omega_n^2 = 0
\end{aligned}
```

When we match the coefficients, we get:

```math
\begin{aligned}
\omega_n = \sqrt{\frac{K_p}{4I_{zz}}}
\end{aligned}
```

```math
\begin{aligned}
\zeta_{CL} = \frac{K_d}{2I_{zz}\omega_n} = \frac{K_d}{2\sqrt{K_p I_{zz}}}
\end{aligned}
```

*Note :- $`\omega_n`$ and $`\zeta_{CL}`$ are the closed loop natural frequency and damping ratio. Together, they’re a convenient way to describe (and design for) the closed loop transient response! things like rise time,settling speed and overshoot are connected to them, for as long as the closed loop behavior is well approximated by a dominant second order response.*

Given a desired natural frequency $`\omega_n`$ and damping ratio $`\zeta_{CL}`$, we can directly translate our performance specs into controller gains as:

```math
\begin{aligned}
\boxed{K_p = 4\omega_n^2 I_{zz}}
\end{aligned}
```

```math
\begin{aligned}
\boxed{K_d = 2\zeta_{CL}\omega_n I_{zz}}
\end{aligned}
```

However, we can’t just choose any $`\omega_n`$ we want. In principle we can set it wherever, but in practice we need to keep $`\omega_n`$  away from our plant’s resonant modes. otherwise may excite those resonances and end up with extra peaking, vibration, noise sensitivity, or even instability.

*Note :- for most practical designs (especially when the closed loop looks roughly second order), $`\omega_n`$ is a good proxy for closed loop bandwidth. eventhough its not an exact match, but it's usually the right handle on how fast the loop responds.*

Now, we know that our [sensitivity fucntion](https://en.wikipedia.org/wiki/Sensitivity_(control_systems)) can be written as:

```math
\begin{aligned}
S(s) = \frac{1}{1+ L(s)}
\end{aligned}
```
So if $`|L(j\omega_{mode})| > 1`$, the controller has significant gain at the modal frequency and will interact with (and potentially destabilise) the flexible dynamics.

To keep our closed loop bandwidth well below the resonance frequency, i propose:


```math
\begin{aligned}
f_{BW} < \frac{f_{mode,1}}{2.5}
\end{aligned}
```

where $`f_{mode,1}`$ is the first resonannce frequency.

*Note:- The “2.5×” separation is a bit conservative, but with modes this lightly damped ($`\zeta \approx 0.02`$) I’d rather play it safe. That said, I might be able to push the bandwidth depending on how well the trajectory shaping (feedforward) keeps energy out of the first resonance. If the reference the feedback loop has to track has low spectral content near the first resonant mode, we can usually get away with a bit more bandwidth. And if we’re lucky, the control design might add some apparent damping around the resonance! but with low bandwidth, any damping injection there will likely be limited. SO no hopes of active damping here in the traditional sense atleast.*

In the case our spacecraft:

- First mode: $`f_1 = 0.4`$ Hz
- Maximum bandwidth: $`f_{BW,max} = 0.4/2.5 = 0.16`$ Hz
- Corresponding $`\omega_n = 2\pi \times 0.16 \approx 1.0`$ rad/s

Now that we have our bandwidth, we can back calculate our gains. But first we consider our effective inertia abut the Z axis, so we have:

```math
\begin{aligned}
I_{eff,zz} = I_{hub,zz} + \sum_1 ^4 M_{modal} r_i^2
\end{aligned}
```
where $`M_{modal}`$ is the modal point masses of our solar array (5 $`kg`$ each per mode) and  $`r_i`$ is the ditance of the  $`i^{th}`$ modal mass from the hub's center.

*Note :- You can find the schematics in the first section should you need a visualisation*

So we have:
```math
\begin{aligned}
I_{eff,zz} = 600 + 325 = 925 kg\cdot m^2 
\end{aligned}
```

Since our $`\omega_n \approx 1.0 \text{ rad/s}`$ (from $`2\pi \times 0.16`$):

```math
\begin{aligned}
\boxed{K_p = 4 \times 1.0^2 \times 925 = 3700\ \text{N}\cdot\text{m}}
\end{aligned}
```

With the proportional gain set, we choose a closed loop damping ratio of $`\zeta_{CL}=0.9`$. This corresponds to a phase margin of about $`73^\circ`$ for a standard second order loop, which sits right in our $`70^\circ`$ to $`75^\circ`$ robustness target.

Using the gain relation derived above,

```math
\begin{aligned}
\boxed{K_d = 2\zeta_{CL}\omega_n I_{zz}}
\end{aligned}
```

we obtain

```math
\begin{aligned}
\boxed{K_d = 2 \times 0.9 \times 1.0 \times 925 = 1665\ \text{N}\cdot\text{m}\cdot\text{s}}
\end{aligned}
```
*Note: This derivation uses a second order approximation, so expect some follow up tuning in simulation. You can do this manually  or set up an optimizer to selcet PD gains that satisfy the closed loop requirements (objective,constraints...etc). i used an optimizer to tune $`K_p = 3.74\times10^3`$ and $`K_d = 1.67\times10^3`$ and validated the outcome in simulation. (tools like copilot ot chatgpt can help with grunt work and setups, but the specs and validation are still on us.)*


Now with our gains finalised, we look at the bode plot of our loop transfer function $`L(s)`$ :

<div style="display: flex; justify-content: center; gap: 10px;">

<image src = "plots\image-4.png" width = 500>

</div>

From the Bode plot, we observe that the loop gain satisfies $`|L(j\omega)| > 1`$ for $`\omega \le 0.3 Hz`$, which supports good low frequency tracking of the shaped reference spectrum. Note that the first resonant mode lies above the $`0 dB`$ line? this is indicating that the loop has significant interaction with that mode. Next, we evaluate whether this interaction excites the resonance or provides additional damping.

### 6.4 Closed Loop System Characteristics

In our prevous sections we have repeatedly introduced our plant model in the frequency domain:

```math
\begin{aligned}
G_{flex}(s) = \frac{1}{4I_{eff}s^2} \cdot \prod_{i=1}^{N_{modes}} \frac{s^2 + 2\zeta_i\omega_i s + \omega_i^2 + \Delta_i}{s^2 + 2\zeta_i\omega_i s + \omega_i^2}
\end{aligned}
```

and we have also derived the dependancy of our closed loop damping ration on our derivative gain given as:

```math
\begin{aligned}
\boxed{K_d = 2\zeta_{CL}\omega_n I_{zz}}
\end{aligned}
```

Now we will see if our derivative term is actually injecting viscous damping at $`L(j\omega_{mode1})`$ since $`|L(j\omega_{mode1})|> 1`$.

For a loop transfer fucntion $`L(j\omega)`$ one can easily determine the damping (or anti damping) of a particular frequency by considering the real part! i.e :


```math
\begin{aligned}
\Re (L(j\omega)) = |L(j\omega)|cos(\angle L(j\omega))
\end{aligned}
```

So, if:

```math
\begin{aligned}
\Re (L(j\omega))>0 \rightarrow damping
\end{aligned}
```

```math
\begin{aligned}
\Re (L(j\omega))<0 \rightarrow excitation
\end{aligned}
```

Since our first resonant mode has a loop gain of $`3.5 dB`$ and a phase of $`-66^\circ`$ , we have:

```math
\begin{aligned}
|L (\omega_{mode1})| \approx 10^{\frac{3.5}{20}} \approx 1.50
\end{aligned}
```

```math
\begin{aligned}
\Re(L(\omega_{mode 1})) \approx 1.5cos(-66^\circ) \approx 0.61 > 0 
\end{aligned}
```

Therefore, at the first resonance frequency, the loop exhibits a damping  contribution, since the in phase component satisfies $`\Re\{L(j\omega_{mode1})\} > 0`$ (using the standard rate loop interpretation).

*Note: if you are wondering about the second resonance mode, $`\Re\{L(j\omega_{mode2})\} \approx 0.788`$ (with $`|L|=-0.2\,\mathrm{dB}`$ and phase $`-36.3^\circ`$).*

This also highlights that setting the closed loop bandwidth below the first resonance does not guarantee zero interaction  if the open loop gain remains non negligible at $`\omega_{mode}`$, the controller can still influence the mode, potentially exciting them depending on their phase in the rate loop.


Now, we look into the stability of our closed loop system. Looking back at the Nyquist satibility criteria introduced in section 6.1, we assess our nyquist plot given below:



<div style="display: flex; justify-content: center; gap: 10px;">

<image src = "plots\image-5.png" width = 500>

</div>

From the Nyquist plot, we immediately see that the closed loop system is stable, since the open loop locus does not encircle the critical point $`(-1, 0)`$. The plot also confirms that the phase margin target is preserved, which is consistent with the derivative gain back calculation used in the design. Finally, the two rapid phase rotations are clearly visible, corresponding to the sharp phase transitions associated with the first and second resonant modes.

*Note :- you can verify the phase transition from our bode plot of $`L(j\omega)`$ given in the previous section.*

Now that we’ve checked the Nyquist plot and convinced ourselves the closed loop is stable (with a decent phase margin), the next step is to look at how the loop behaves for tracking, disturbance rejection, and noise rejection. The standard way to do that is by plotting the sensitivity functions: the sensitivity $`S(j\omega)`$ (disturbance rejection) and complementary sensitivity $`T(j\omega)`$ (tracking / noise shaping).

<div style="display: flex; justify-content: center; gap: 10px;">
  <image src="plots\image-6.png" width=500>
</div>

From the sensitivity plot, it looks like we get strong disturbance rejection up to about $`0.25\,\text{Hz}`$ (highlighted in green). Around the two resonance frequencies, $`S(j\omega)`$ shows dips (roughly $`-5dB`$ in our plot). That’s consistent with the loop having meaningful authority at those frequencies (and in our case, with the added damping from rate feedback helping to keep the resonances under control).

Also notice the dip at the first resonance is deeper than the one at the second resonance? that typically lines up with higher loop gain at the first mode (you can cross check this directly in the bode plot of the loop transfer function).

One thing we need to keep an eye on is the anti resonance region. you can see $`|S(j\omega)|`$ rises there, meaning disturbances near the anti resonance are rejected less effectively. Practically, that’s a bit risky because a disturbance around the anti resonance frequency can excite the array dynamics while producing relatively little motion at the hub! so it can shake our arrays without an obvious hub response.

*Note:- Honestly, I find anti resonances even scarier than resonances. Around those frequencies, the hub response can look harmlessly small, even while our arrays are moving a lot. That means a disturbance or noise near that frequency could excite array vibration without it showing up clearly in the attitude signal! So we should make sure our reference shaping  keeps energy out of those anti resonance bands as well.*

As we move to higher frequencies, the loop doesn’t reject disturbances very well (the yellowish orange looking region), and that’s pretty expected. In those bands the loop gain is low, so $`S(j\omega)\approx 1`$, meaning disturbances mostly pass through. The upside is that low loop gain makes $`T(j\omega)\approx 0`$, so measurement noise is attenuated. In other words, you generally can’t get strong disturbance rejection and strong noise rejection in the same frequency range. Sadly, it’s an inherent trade off (since $`S+T=1`$ for unity feedback).


Moving on to our complementary sensitivity function $`T(j\omega)`$, one thing jumps out immediately, $`|T(j\omega)|>1`$ at frequencies leading up to about $`0.2\text{ Hz}`$. That peaking can be great for tracking in that band, since the output can follow reference components there very tightly ( in magnitude terms, even with gain greater than unity!).

The catch is that it’s a bit of a double edged sword. Any signal that reaches the output through $`T(j\omega)`$ gets the same treatment, so measurement side effects like slow sensor drift or low frequency bias content in this band can also be passed through more strongly (and potentially amplified where $`|T|>1`$.). [Peaking](https://web.stanford.edu/class/archive/ee/ee392m/ee392m.1056/Lecture8_SISO_Design.pdf) is also a reminder that we are trading away some robustness, since it often corresponds to a lightly damped closed loop feature.

If you look at the $`-3 dB`$ line on the complementary sensitivity plot, you can also see that our closed loop bandwidth shifted compared to our calculation in the previous section. This was intentional because the structural resonance modes are now being (light to moderately) damped, we were able to push the bandwidth forward with the intent i described in the previous section. That, in turn, is why relaxing the earlier $`2.5\times`$ constraint (which we already noted was conservative) was a deliberate choice to reduce sensitivity to disturbances around the anti resonance frequencies.

Looking at the higher frequency part of our complementary sensitivity $`T(j\omega)`$ plot, we can see that sensor noise (or other measurement artifacts) around the first resonant mode is not attenuated very well as $`|T(j\omega)|`$ sits only slightly above the $`-3 dB`$ line there (so we are getting less than about 3 dB of noise rejection in that neighborhood). Past that region, the high frequency roll off becomes more regular, and the noise attenuation is more consistent.

*Note :- You might also notice a “nice” dip (extra attenuation) near an anti resonance in $`T(j\omega)`$. That dip is not automatically a win for tracking, it usually means the plant has a transmission zero there, so reference to output motion is intrinsically small at that frequency (i.e. tracking is poor in that band). Physically, that can happen because the actuator effort is going into bendign of the solar arrays rather than rigid body rotation at the measured output, so the output looks quiet (observability nullspace) even though the structure can be getting excited and internal loads and motion can increase!*

By looking at the sensitivity $`S(j\omega)`$ and complementary sensitivity $`T(j\omega)`$, we can answer the big picture questions:

- How well does our feedback loop reject disturbances and plant uncertainty across frequency?
- How well does our loop track the reference, and how much measurement noise gets through?

That already tells us a lot about the closed loop behavior. But for a safety critical system like a satellite, I also want more concrete  questions answered:

- If a disturbance torque hits the spacecraft, how much pointing error do we actually see at the output? (For input torque disturbances, the relevant path is typically $`G(j\omega)S(j\omega)`$.)
- if the gyro rate measurement has noise (or drift,ripple), how much pointing error does that create?


I’m asking these because the requirement in Section 1 is stated in time domain terms where we need pointing stability < 5 arcsec within 5 seconds. By quantifying how disturbances and sensor noise map into pointing error and commanded torque, we can directly check whether the design is consistent with that requirement and see what the real margins are !.


To understand how input disturbance torques map into pointing error, we look at the closed-loop transfer function $`G(j\omega)S(j\omega)`$, shown below:

<div style="display: flex; justify-content: center; gap: 10px;">
  <image src="plots\image-7.png" width=500>
</div>

From the plot, the first thing that jumps out is that below about $`0.1\,\text{Hz}`$ the curve is fairly flat at roughly $`-25\,\text{dB}`$. Converting that to linear units:

$`10^{-25/20} \approx 0.056^\circ/N\cdotp m \approx 202 arcsec/N·m`$.

At the first resonance (around $`0.4Hz`$), the magnitude is closer to $`-40dB`$, which corresponds to:

$`10^{-40/20} \approx 0.01^\circ/N·m \approx 36arcsec/N\cdotp m`$.

Beyond that, the response continues to roll off with frequency. The picture is pretty clear here! the worst case pointing sensitivity to input torque disturbances occurs at low frequencies (roughly $` \le 0.1\,\text{Hz}`$), while higher frequency torque disturbances does not influence our pointing as much.

*Note: If this seems a bit inconsistent with what we saw in $`S(j\omega)`$, it’s because this plot is a answering slightly different question. The transfer $`G(j\omega)S(j\omega)`$ describes how an input disturbance torque (added at the plant input) shows up as pointing error. In contrast, the sensitivity function $`S(j\omega)`$ by itself corresponds to disturbances injected at the output. So there iss no real contradiction! we are just looking at different disturbance locations in the loop.*

Next we check much does our gyro rate measurement noise affect our pointing error, and for that we look at the transfer function $`\frac{G(j\omega)K_d}{1+L(j\omega)}`$. Notice how our derivative gain directly influencing the effect of measurement noise on pointing error? we will see just how much the influence is in the bode plot below:



<div style="display: flex; justify-content: center; gap: 10px;">
  <image src="plots\image-8.png" width=500>
</div>

Well, right off the bat we see a huge gain at low frequencies (below 0.1 Hz) of about $`40\ \mathrm{dB}`$ (i.e roughly a $`100\times`$ linear gain, since $`20\log_{10}(100)=40`$). 

So intuitively, a rate bias like error of:

- $`10^{-6}\ \mathrm{rad/s} \approx 0.36\ \mathrm{arcsec}`$
- $`10^{-5}\ \mathrm{rad/s} \approx 3.6\ \mathrm{arcsec}`$
- $`2\times10^{-5}\ \mathrm{rad/s} \approx 7.2\ \mathrm{arcsec}\ \rightarrow`$ requirement violation

So, slow gyro drift,bias (i.e low frequency rate noise) will get converted into real pointing error pretty efficiently, sadly.

*Note :- We can reduce the effects of low frequency gyro measurement artifacts by reducing the derivative gain (or second order low passing the derivative term). However, by doing so we reduce damping of resonant modes and increase the closed loop settling time. So it’s a real design trade off, at least in my opinion.*

## 7 FeedForward Control Design

In the previous sections, we established the limitations of our feedback controller. With those constraints in mind, this section focuses on designing a feedforward strategy through trajectory shaping. The primary goal is to avoid exciting resonant modes (and anti resonance) while making it easier for the feedback loop to track the commanded instanteneous trajectory.

We achieve this by deliberately shaping the reference trajectory so that its spectral content is small at, and especially beyond, the first anti resonant mode. Referring back to the complementary sensitivity function from the previous section, we observed that the feedback controller provides strong tracking performance up to about 0.2 Hz. Therefore, we try to design the reference trajectory to contain minimal spectral content above the feedback bandwidth (also adhereing to our unrealistic slew requirement!).

*Note :- For this project, the anti resonance does not significantly hurt our pointing stability, because energy at that frequency primarily goes into shaking the arrays while producing very limited hub (bus) oscillations. So our comet imaging should not be affected by it. However, we will still avoid exciting it, since in real life excessive array motion could lead to structural damage, and I will take that into account when shaping our reference trajectory.*

Before we start with trajectory shaping, I want to briefly cover a bit of theory to help motivate my approach. Now that being said,lets consider a rectangular time domain [window function](https://en.wikipedia.org/wiki/Window_function) with a window length of 1 second. When we represent this window in the frequency domain, it becomes a [sinc function](https://en.wikipedia.org/wiki/Sinc_function), as shown below:

<div style="display: flex; justify-content: center; gap: 10px;">
  <image src="plots\image-10.png" width=500>
</div>

Now, if you look at the region highlighted by the two arrows, you can see spectral nulls or in other words, frequencies where the spectral magnitude (and therefore the power) is vanishingly close to zero. To make this observation more rigorous, lets derive this behavior mathematically:

Consider a unit amplitude rectangular window of duration $`T`$, centered at $`t=0`$:

```math
\begin{aligned}
w(t) &=
\begin{cases}
1, & |t|\le \tfrac{T}{2} \\
0, & \text{otherwise}
\end{cases}
\end{aligned}
```
using the continuous time Fourier transform [(CTFT)](https://en.wikipedia.org/wiki/Fourier_transform) definition:

```math
\begin{aligned}
W(\omega) &= \int_{-\infty}^{\infty} w(t)\,e^{-j\omega t}\,dt
\end{aligned}
```

since $`w(t)`$ is only nonzero on $`\left[-\tfrac{T}{2},\,\tfrac{T}{2}\right]`$, we have:

```math
\begin{aligned}
W(\omega) &= \int_{-T/2}^{T/2} e^{-j\omega t}\,dt
\end{aligned}
```

evaluating the above integral, we get:

```math
\begin{aligned}
W(\omega)
&= \left[\frac{e^{-j\omega t}}{-j\omega}\right]_{-T/2}^{T/2} \\
&= \frac{1}{-j\omega}\Big(e^{-j\omega T/2}-e^{+j\omega T/2}\Big)
\end{aligned}
```

now we use the identity $`e^{-ja}-e^{ja}=-2j\sin(a)`$ to yeild:

```math
\begin{aligned}
W(\omega)
&= \frac{1}{-j\omega}\Big(-2j\sin(\omega T/2)\Big) \\
&= \frac{2\sin(\omega T/2)}{\omega}
\end{aligned}
```

now we rewrite it in a standard sinc form:

```math
\begin{aligned}
W(\omega)
&= T\,\frac{\sin(\omega T/2)}{\omega T/2}
\end{aligned}
```

we define, $`\mathrm{sinc}(x)=\dfrac{\sin(x)}{x}`$. Then:

```math
\begin{aligned}
W(\omega) &= T\,\mathrm{sinc}\!\left(\frac{\omega T}{2}\right)
\end{aligned}
```

with this, we can write the magnitude spectrumas :

```math
\begin{aligned}
|W(\omega)| &= T\left|\mathrm{sinc}\!\left(\frac{\omega T}{2}\right)\right|
\end{aligned}
```

looking at this magnitude spectrum, we can see that our spectral nulls lie where the numerator are zero! so we equate it to zero, and yeild:

```math
\begin{aligned}
\sin(\omega T/2) &= 0
\end{aligned}
```
and this happens when:

```math
\begin{aligned}
\frac{\omega T}{2} &= k\pi,\quad k=\pm 1,\pm 2,\pm 3,\ldots
\end{aligned}
```

so our null locations in rad/s are:

```math
\begin{aligned}
\omega_k &= \frac{2\pi k}{T}
\end{aligned}
```

converting it to Hz using $`\omega = 2\pi f`$:

```math
\begin{aligned}
2\pi f_k &= \frac{2\pi k}{T} \\
f_k &= \frac{k}{T}
\end{aligned}
```

so for our plot where the window length $`T=1`$, we have:

```math
\begin{aligned}
f_k &= k\ \text{Hz},\quad k=\pm 1,\pm 2,\pm 3,\ldots
\end{aligned}
```

So the first few spectral nulls occur at 1 Hz, 2 Hz, 3 Hz,... and so on.

Notice what happened here? the spectral nulls occured at frequency $`\frac{1}{T}`$ (where $`T`$ is our window length), and its [Harmonics](https://en.wikipedia.org/wiki/Harmonic)!

To illustrate where I’m going with this, let’s consider a superimposed sinusoidal signal composed of the frequencies $`1 \mathrm{Hz},2.5\ \mathrm{Hz}, 5 \mathrm{Hz}, and 10\mathrm{Hz}`$. We then convolve this signal with our rectangular window (the one plotted above), and observe the output shown below:


<div style="display: flex; justify-content: center; gap: 10px;">
  <image src="plots\image-11.png" width=500>
</div>

From these figures, we can quickly see that after convolution the PSD shows vanishingly small power at frequencies other than $`2.5\ \mathrm{Hz}`$. This is exactly what we expect for a rectangular window of length $`1\ \mathrm{s}`$, the spectral nulls occur at integer multiples of $`1\ \mathrm{Hz}`$. If we also want to null $`2.5\ \mathrm{Hz}`$, we can simply increase the window length to $`2\ \mathrm{s}`$, which places nulls every $`0.5\ \mathrm{Hz}`$.

This observation forms the basis of our setpoint shaping approach. We avoid exciting our solar array resonance (and anti resonance) by convolving the reference signal with  window functions that place spectral nulls at frequencies that are critical to us. Pretty cool fourier transform trick, right?

Our torque feedforward can be computed as:

```math
\begin{aligned}
\tau_{ff} = I_{eff,zz} \alpha(t)
\end{aligned}
```
where $`\tau_{ff}`$ is the feedforward torque, $`I_{eff,zz}`$ is the effective inertial along the Z axis, and $`\alpha(t)`$ is our acceleration profile which we will be shaping!

We begin by convolving the base acceleration pulse:

```math
\begin{aligned}
\alpha_0(t) = A \cdot \text{rect}\left(\frac{t - T_b/2}{T_b}\right)
\end{aligned}
```

with a rectangular window of duration $`T_1 = 1/f_1 = 2.5 s`$ this would place the spectral nulls at our first resonant mode($`0.4 Hz`$) and its harmonics.



 So our convolved acceleration profile can be written as:

 ```math
\begin{aligned}
\alpha_1(t) = \alpha_0(t) * w_{T_1}(t)
\end{aligned}
```

where the window is:

```math
\begin{aligned}
w_{T_1}(t) = \frac{1}{T_1} \cdot \text{rect}\left(\frac{t - T_1/2}{T_1}\right)
\end{aligned}
```

*Note :- convolution with a rectangular window is equivalent to a moving average. The sharp step edges of the base pulse get smoothed out into linear ramps (we are limiting the jerk!). The result is a trapezoidal acceleration profile*.

So the first convolution gave us a [jerk](https://en.wikipedia.org/wiki/Jerk_(physics)) limited profile, thus smoothing out the discontinous initial acceleration base pulse. Subsequently, we will proceed with the second convolution to [snap](https://en.wikipedia.org/wiki/Fourth,_fifth,_and_sixth_derivatives_of_position) limit our acceleration profile (and we place the spectral nulls at the second resonant mode). The convolution with a second rectangular window of duration $`T_2 = 1/f_2 = 0.769s`$ yeilds:


```math
\begin{aligned}
\alpha_2(t) = \alpha_1(t) * w_{T_2}(t)
\end{aligned}
```

For a rest to rest manoeuvre (starting and ending at zero velocity), we need to decelerate after accelerating. Hence the full profile is created by mirroring and negating:

```math
\begin{aligned}
\alpha(t) = \begin{cases}
\alpha_2(t) & 0 \le t < T_{half} \\
0 & t = T_{half} \\
-\alpha_2(T_{total} - t) & T_{half} < t \le T_{total}
\end{cases}
\end{aligned}
```


Once we have $`\alpha(t)`$, we integrate to get velocity and position:

```math
\begin{aligned}
\omega(t) = \int_0^t \alpha(\tau) \, d\tau
\end{aligned}
```

```math
\begin{aligned}
\theta(t) = \int_0^t \omega(\tau) \, d\tau
\end{aligned}
```

In discrete time:

```math
\begin{aligned}
\omega[k] = \sum_{i=0}^{k} \alpha[i] \cdot \Delta t
\end{aligned}
```

```math
\begin{aligned}
\theta[k] = \sum_{i=0}^{k} \omega[i] \cdot \Delta t
\end{aligned}
```

*Note :- These two are our instantenous reference trajectory that our feedback tried to follow. In our codes, this theta is converted into MRP vector (as our feedback law uses MRP error)*



The raw trajectory won't hit our target angle $`\theta_{final}`$ exactly. So, we compute a scale factor:

```math
\begin{aligned}
\text{scale} = \frac{\theta_{final}}{\theta_{raw}(T_{total})}
\end{aligned}
```

Then we scale all profiles:

```math
\begin{aligned}
\theta(t) &\leftarrow \text{scale} \cdot \theta(t) \\
\omega(t) &\leftarrow \text{scale} \cdot \omega(t) \\
\alpha(t) &\leftarrow \text{scale} \cdot \alpha(t)
\end{aligned}
```

 Scaling preserves the spectral zero locations! multiplying by a constant in time domain just multiplies the fourier transform by the same constant, it doesn't shift where the zeros are.


Finally, the feedforward torque is simply Newton's second law for rotation:

```math
\begin{aligned}
\tau_{FF}(t) = I_{eff,zz} \cdot \alpha(t)
\end{aligned}
```
*Note :- fourth order setpoint shaping is not a cheat code! the convolution process spreads the acceleration profile over a longer time window, which means we face a fundamental tradeoff:*

```math
\begin{aligned}
\theta = \int_0 ^T \int_0 ^ t \alpha(\tau) d\tau dt 
\end{aligned}
```

*for a fixed rotation angle $`\theta`$, the double integral of acceleration is constrained. Thus,smoothing the profile (via convolution) reduces the peak acceleration but extends the duration. Hence the penality is:*

- *If we constrain the move time, we pay with higher peak torque*
- *If we constrain the peak torque, we pay with longer move time*

*In our design, the shaping adds approximately 3.3 seconds of overhead per half manoeuvre. So sor a 30 second slew, that's about 11% of the budget spent on smoothness rather than slewing!*

Now with that being said, we look at a comparion plot between a bang bang acceleration profile, and our fourth order shaped profile, used for the same $`180^\circ`$ slew manoeuvre:

<div style="display: flex; justify-content: center; gap: 10px;">
  <image src="plots\image-12.png" width=500>
</div>

From this plot, we can see that the fourth order shaped profile has much less spectral power above 0.4 Hz than the bang bang acceleration profile. In practice, this means the shaped reference trajectory contains much less energy in that range, so it is much less likely to excite the solar array’s flexible dynamics during the slew manoeuvre.

*Note :- In real applications, achieving zero solar array vibration is essentially impossible. Unmodelled actuator dynamics, parameter uncertainty, and other disturbances can still inject energy into lightly damped resonant modes, so some level of residual vibration during (and after) the manoeuvre should be expected.*

With the trajectory now shaped, we can visualise the reference position, velocity, acceleration, jerk, and snap as shown below:

<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="plots\image-13.png" width="500">
</div>

From these plots, we can see that the instantaneous reference signals tracked by the feedback controller are significantly smoother. However, discontinuities still appear in the snap profile, which implies an infinite crackle (the sixth derivative of position). See: [Fourth, fifth, and sixth derivatives of position](https://en.wikipedia.org/wiki/Fourth,_fifth,_and_sixth_derivatives_of_position).

*Note: This snap discontinuity is of little practical consequence for our application. Attempting to limit crackle by applying an additional convolution would typically introduce a tradeoff either a longer manoeuvre time or a higher peak torque while providing negligible performance improvement in our specific scenario.*

Well, now that we have shaped our trajectory and designed the feedforward, the total torque that is commanded by our controller will be:



```math
\begin{aligned}
\tau_{total} = \tau_{fb} + \tau_{ff} 
\end{aligned}
```
With this, we can represent our control system schematics as below:


<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="plots\image-14.png" >
</div>

## 8 Verification and Validation

In the earlier sections, we mainly focused on building the plant model, linearizing it around our assumptions, designing the control system, and tuning the gains. Now we are going to shift to verification and validation of our design. Basically, making sure the design performs the way we expect and that our assumptions still hold in simulation. We will break this section into three parts:

- Mission run (verification against requirements): We will run the repositioning mission (a 180 deg slew in 30 s) and check whether all requirements are met. We will also monitor the feedback error to see if it grows large enough to invalidate the linearization and feedback control assumptions. From there, we will look at whether the feedforward and feedback torque commands contain significant energy near the resonant and antiresonant modes, and we will examine modal displacement and acceleration. Finally, we will compare all results against a baseline case: the same mission executed with S-curve feedforward.

- Parameter sweep (sensitivity testing): Next, we will run a parameter sweep (one parameter at a time) to identify the conditions under which the mission requirements start to break down. This includes sweeps over inertia variation, damping variation, resonant frequency variations, disturbance levels, and noise levels.

- Monte Carlo (robustness assessment): Lastly, we will run a Monte Carlo simulation with random parameter variations (+-30% around the design values) to estimate how often the requirements are violated under uncertainty.

*Note :- for the verification and validation, we will be using the nonlinear basilisk model*

### 8.1 Mission Run

In this subsection, we simulate the repositioning mission and walk through a comparative analysis of the results. To make it easier to understand how the simulation is put together, here is the simulation architecture we are using:

<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="plots\image-15.png" width="500">
</div>

Now that the simulation architecture is defined, we can evaluate its results with a bit more context.

To kick off the analysis, let us start with the modal displacement and modal acceleration responses, shown below:

<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="plots\mission_vibration.png" width="500">
</div>

If you look at the modal displacement and acceleration plots, you will notice the difference is pretty clear. The fourth order shaped feedforward + PD feedback brings the post slew RMS displacement down from $`0.2534\ \mathrm{mm}`$ (S-curve + PD) to $`0.099\ \mathrm{mm}`$, which works out to a bit over 60% reduction in displacement.

The acceleration improvement is also significant, where the post slew RMS modal acceleration drops from about $`1.561\ \mathrm{mm/s^2}`$ (S-curve + PD) to about $`0.445\ \mathrm{mm/s^2}`$ (fourth order + PD), i.e. roughly a 71.52% reduction.

These two plots are great for getting the overall picture, but our main priority is the vibration after the slew is complete, i.e. the residual vibration, since that falls directly inside the mission's imaging window. Thus, we need to figure out the frequency that contributes the largest amount to these post slew ringdowns, or more precisely we need to identify the lightly damped mode. The clean way to do that is to compute the power spectral density (PSD) and look at it in the plot below:

<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="plots\modal_acceleration_psd.png" width="500">
</div>

From this plot it is more or less obvious that the first mode carries more power in both cases. However, we still see a discrepancy in the ratio of power content between the first and second mode, despite the PD control being identical.

*Note :- post slew manoeuvre, feedback control dominates as the torque from the feedforward is zero*

To clarify, let $`P_1`$ be the first mode peak power and $`P_2`$ be the second mode peak power in the post-slew modal acceleration PSD.

From the computed post slew PSD peaks, we have:

```math
\begin{aligned}
\text{S-curve:}\quad &P_{1,s} \approx -47.84\ \mathrm{dB}, \quad P_{2,s} \approx -71.77\ \mathrm{dB} \\
\text{Fourth order:}\quad &P_{1,4} \approx -57.95\ \mathrm{dB}, \quad P_{2,4} \approx -88.72\ \mathrm{dB}
\end{aligned}
```

Now converting dB differences to linear power ratios:

```math
\begin{aligned}
\frac{P_{1,s}}{P_{2,s}} &= 10^{\frac{P_{1,s,\mathrm{dB}} - P_{2,s,\mathrm{dB}}}{10}}
= 10^{\frac{-47.84 - (-71.77)}{10}}
= 10^{2.393}
\approx 247.25
\end{aligned}
```

```math
\begin{aligned}
\frac{P_{1,4}}{P_{2,4}} &= 10^{\frac{P_{1,4,\mathrm{dB}} - P_{2,4,\mathrm{dB}}}{10}}
= 10^{\frac{-57.95 - (-88.72)}{10}}
= 10^{3.077}
\approx 1193.67
\end{aligned}
```

Equivalently, if we compare second mode power relative to first mode power:

```math
\begin{aligned}
\left(\frac{P_2}{P_1}\right)_s \approx 0.004044,\qquad
\left(\frac{P_2}{P_1}\right)_4 \approx 0.000838
\end{aligned}
```

so the relative second mode content is reduced by:

```math
\begin{aligned}
\frac{(P_2/P_1)_s}{(P_2/P_1)_4}
\approx \frac{0.004044}{0.000838}
\approx 4.83\times
\end{aligned}
```

This is the key point, feedback alone cannot create this difference when the PD law and gains are essentially the same.

So why does the second mode show noticeably lower power than the first when we use the fourth order setpoint shaping? It is because the feedforward command simply does not inject as much energy into that mode in the first place.

Relative to the S-curve profile, the fourth order profile still leaves less residual mode 2 content after the slew. In other words, both references are smooth, but the fourth order shaping does a better job of keeping injected energy away from the structural modes.

So the standout feature of our control approach is not that the PD feedback is doing some magical level of active damping. The real win is that setpoint shaping reduces how much we excite the resonant modes to begin with, which makes the vibrations smaller from the start and easier to manage overall.

And yes, if we are giving credit where it is due, kudos to setpoint shaping.

Now that we have seen the improvement in residual vibrations, the next step is to make sure those gains did not come at the expense of violating the small angle (linearization) assumption used to design the PD feedback controller. A straightforward way to check that is to look at the attitude tracking error during the maneuver:

<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="plots\mission_tracking_error.png" width="500">
</div>

From the absolute tracking error plot, both the S-curve and fourth order cases stay close to the reference trajectory, with the error remaining bounded and relatively small throughout the slew. This suggests that the resulting deviation from the instantaneous reference does not grow large enough to invalidate the linear model assumptions behind the PD controller design.

*Note :- S-curve exhibits a lower peak tracking error compared to the fourth order. This is because S-curve has lower vibrations mid slew compared to the fourth order.*

On another note, if you look closely at the absolute tracking error plot, the S-curve + standard PD case has a clearer oscillatory pattern than the fourth order case. That oscillation lines up with the stronger residual flexible response we saw in the vibration plots.

At the same time, the tracking error also shows a slower, non oscillatory trend in the background. To understand what is driving each part of the response (the oscillatory content versus the slower component), we can look at the power spectral density (PSD) of the absolute tracking error:

<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="plots\mission_tracking_error_psd.png" width="500">
</div>

From this PSD plot, it is clear that the slow trend we noticed in the absolute tracking error (in both cases) is coming from very low frequency content, mainly below about $`0.1\ \text{Hz}`$.

That is exactly what we would expect. If we refer back to the closed loop characteristics in section 6.4, the disturbance torque -> pointing error transfer function shows that the system is most sensitive at low frequencies, particularly around $`0.1\ \text{Hz}`$ and below.

So the PSD is really just confirming the same point, the non oscillatory portion of the tracking error is mainly driven by low frequency input disturbances. In our case, the gravity gradient torque and the SRP torque included in the simulation.

Now that we’ve confirmed the feedback controller stays within the small angle assumptions used in its design, we can move on to evaluating post slew pointing stability.

More specifically, we want to verify that the post slew RMS pointing error remains below 5 arcseconds, as required. To do that, we refer to the pointing error plot below:


<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="plots\mission_pointing_error.png" width="500">
</div>

From the plot above, the fouth order setpoint shaping cuts the post slew RMS pointing error by roughly 64.1%. More importantly, it satisfies the post slew pointing stability requirement, reaching an RMS pointing error of 4.65 arcsec.

What’s especially telling is that this improvement happens without changing the feedback controller. That strongly suggests the main driver here is the trajectory shaping, reducing how much we excite the structural resonant modes in the first place, rather than relying on the closed loop controller to “actively” add damping. For this project, avoiding mode excitation upfront turns out to be the more effective approach!

Now that we’ve established the performance improvements, the next step is to see what they actually mean for the mission,especially for long exposure imaging of the comet.

To do that, we run a comet camera simulation that models the image blur caused by residual vibrations. The vibration response is converted into angular jitter using an assumed lever arm, and that jitter is then applied during the exposure to estimate the resulting blur.

We then compare the image resolution and the jitter PSD for the different cases, as shown in the results below:

<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="plots\comet_blur_comparison_psd_check.png" width="500">
</div>


From this we understand that the fourth order setpoint shaping improves the image resolution by reducing the blur from 10 pixes to 3.3 pixels yeilding a 70% imporvement in blur reduction!

### 8.2 Parameter Sweep

In the previous section, we ran a full mission simulation to verify that our design meets the requirements and that the assumptions we made during modeling and controller design still hold in simulation. In other words, that section answered the question,“Does my design work?”

This section tackles a different set of questions “When does the design stop meeting the requirements?” and “Which parameter variations hurt performance the most?”  

To answer those, we will run a set of simulations using a parameter sweep, where we intentionally under and overestimate key model parameters. This lets us quantify how sensitive the design is to uncertainty and identify which parameters are the most critical drivers of requirement violations.


Since the solar array vibrations directly affect post slew pointing stability, we will start this sensitivity study by varying our modal frequency estimate to see whether the design is robust to changes in the structure’s natural frequency.

The motivation is straightforward, in orbit effects like thermal expansion and other structural changes can shift modal frequencies over time. Modeling every possible source of variation is outside the scope of this project, so instead we apply a simple but meaningful test that vary the estimated modal frequency by ±30% and compare how the S-curve profile and the fourth order shaped profile perform.

We will evaluate performance using three main metrics:

- RMS vibration
- RMS pointing error
- Peak torque

*Note: Earlier in the project, we assumed actuator constraints were out of scope. Even so, i am including peak torque here as an additional comparison metric, since it helps highlight practical trade offs between the approaches.*

With that said, we observe the results of our modal sweep below:



<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="plots\mc_sweep_modal_frequency.png" width="500">
</div>

From these plots, it is clear that fourth order setpoint shaping outperforms the S-curve when the modal frequency estimate is within about ±20% of the nominal value.

That said, performance degrades more noticeably when the true natural frequency is lower than expected (i.e., when we have overestimated the mode). The reason is tied to how the fourth order shaping works; it effectively “nulls” vibration starting at the first frequency targeted by the shaping window (through convolution with jerk and snap limited windows). Any excitation below that first targeted frequency isn’t nulled .so if the real mode shifts downward, it can fall into a range where the shaper is no longer doing much.  

This also explains the asymmetry seen in the sweep, overestimating the modal frequency tends to hurt more than underestimating, since underestimation still places the true mode at or above the frequencies the shaper is designed to attenuate.

On the torque side, we only see minor changes in both peak torque and RMS torque across the frequency sweep. That said, the S-curve generally shows a different torque trade off compared to the fourth order shaper. based on the plots, it tends to demand higher peak torque, while the RMS torque behavior differs depending on the specific case.

 *Note :- A proper feasibility study that includes actuator saturation and power/energy budgeting would be needed to make a strong statement about which profile is more energy efficient. But purely in terms of vibration suppression, the fourth order setpoint shaping performs very well as long as the modal frequency estimation error stays within roughly 20%.*

Next, we run a sensitivity study on modal damping ratio.

In space, the primary source of natural damping comes from material/structural damping (as discussed in Section 1). The damping ratio determines how quickly an oscillation dies out. For a lightly damped structure like our solar arrays, getting this parameter wrong can matter especially for post slew pointing stability, because lower than expected damping means the vibrations ring down more slowly.

To evaluate how sensitive our performance is to this uncertainty, we vary the modal damping ratio by ±30% and then compare the resulting changes in vibration suppression and pointing performance, as shown below:

<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="plots\mc_sweep_modal_damping.png" width="500">
</div>

From these plots, the nearly flat curves make the takeaway pretty clear that varying the damping ratio by ±30% doesn’t produce a noticeable change in performance.

That result makes sense when you look at the absolute numbers. With nominal damping ratios of 0.02 and 0.015, a ±30% variation only shifts them to:

- Mode 1: 0.014 to 0.026
- Mode 2: 0.0105 to 0.0195

Those ranges are still very small in absolute terms, so the change in damping isn’t large enough to either positively or negatively affect the vibration or pointing performance in our simulations.



The next set of sweeps focuses mainly on the feedback loop, so we expect the feedforward component to have only a limited effect, especially at the lower frequencies we care about here. Because of that, we would also expect both versions of our control architecture to show very similar behavior in this part of the study.

Based on the closed loop characteristics from the earlier section, we can use the rate noise → pointing error transfer function to guide our intuition. It shows that low frequency noise or drift in the gyro rate measurement tends to produce the largest pointing error, since the controller responds strongly to those components.

To validate that behavior in simulation, we inject a sinusoidal rate noise and sweep its frequency from 0.1 Hz up to 50 Hz (the Nyquist frequency). We then track how the sweep impacts our performance metrics, as shown below:

<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="plots\mc_sweep_noise_level.png" width="500">
</div>

Just as expected, at low frequencies the two cases behave almost identically; the performance trends are essentially the same.

Things start to separate around the ~1.3 Hz region. If we go back to the rate noise → pointing error transfer function, that is roughly where the response drops below 0 dB. In other words, beyond about 1.3 Hz, the rate noise is no longer strongly amplified into pointing error by the feedback loop. From that point onward, the overall behavior is increasingly shaped by what the feedforward/trajectory is doing rather than by the feedback response to measurement noise.

Since we have already seen that fourth order setpoint shaping is better at avoiding excitation of the structural resonances, it makes sense that it maintains a lower RMS pointing error, and as a direct consequence, a lower RMS vibration from about 1.3 Hz and above.

One interesting thing shows up when you look at the peak torque and RMS torque plots. You will notice that whenever the RMS vibration and RMS pointing error spike (i.e., near resonance), the peak torque and RMS torque actually dip.

This makes sense if we relate it back to the closed loop sensitivity behavior discussed in Section 6.4. In those frequency ranges, the closed loop system is less responsive to certain disturbance/noise components coming through the measurement path. As a result, the controller doesn’t “fight” those components as aggressively, and the commanded torque around those frequencies ends up being lower compared to frequency ranges where the loop has higher sensitivity.

*Note :- we can verify this by referring back to the sensitivity fucntion in section 6.4*

In the same way, we also run a sweep on input disturbances and track how they impact our performance metrics, as shown below:

<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="plots\mc_sweep_disturbance_frequency_targeted.png" width="500">
</div>

looking at this plot, we can quickly see that low frequency disturbances contribute to the largest RMS pointing error and RMS vibrations. This behavior is echoed from the disturbance torque to pointing error transfer function given in the closed loop characteristics section (6.4).


*Note :- this implies that gravity gradiant torques, SRP torques and other low frequency torque affects out pointing stability the most*

### 8.3 Monte Carlo Simulation

 In the previous section, we looked at how individual parameter changes affect post slew pointing stability and vibration suppression. Those sweeps are useful for building intuition, but they don’t try to reflect a real mission where multiple parameters can vary at the same time, in different directions.

That is exactly what this Monte Carlo simulation is meant to capture. Here, we randomly vary each parameter within ±20% of its nominal value and run 500 trials in total. We then evaluate the performance of both control architectures, with the main focus on requirement compliance.

Each simulation runs for 90 seconds. The slew ends at t = 30 s, and we compute RMS metrics over the post slew window to check for requirement violations:

- Pointing stability: RMS pointing error must remain below 7 arcsec over the 60 s after the slew  


After identifying any violations, we will investigate why  they happen by correlating the outcomes with the randomly perturbed parameters.

The Monte Carlo results are shown below:

<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="plots\monte_carlo_comparisons_box.png" width="500">
</div>

<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="plots\monte_carlo_post_slew_pointing_box.png" width="500">
</div>

From our monte carlo results, we can clearly see that the fourth order acheived lower pointing error, RMS vibration , and peak torque. However, the fourth order required higher RMS torque than the S-curve profile. on top of these imporvements, the fourth order violations of our mission RMS pointng error requirement was about 13% of the total run, in comparison to 100% of the total run by the S-curve profile. This means that for the 500 runs in the monte carlo, S-curve did not meet the mission RMS pointing error requirement in any of the runs!

Now that we’ve confirmed the fourth order setpoint shaping provides better pointing stability and vibration suppression than the S-curve, the next step is to understand, which uncertain parameters are driving the RMS pointing error for each control architecture.

To do that, we look at the parameter impact histogram below, which shows how strongly each parameter variation contributes to the RMS pointing error:


<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="plots\monte_carlo_post_slew_pointing_factor_hist_fourth_standard_pd.png" width="500">
</div>

<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="plots\monte_carlo_post_slew_pointing_factor_hist_s_curve_standard_pd.png" width="500">
</div>


Each panel in these figures corresponds to one uncertain parameter from the Monte Carlo run. For that parameter, the histogram compares two groups of simulations:

1. a low quartile group, where the parameter is in the lowest 25 percent of sampled values  
2. a high quartile group, where the parameter is in the highest 25 percent of sampled values

The horizontal axis is post slew RMS pointing error in arcsec, and the vertical axis is the number of Monte Carlo runs in each pointing error bin.

So the interpretation is straightforward. If the high quartile distribution shifts to the right relative to the low quartile distribution, increasing that parameter tends to worsen pointing stability. If it shifts to the left, increasing that parameter tends to improve pointing stability.

To make this quantitative, we use the median shift between the two groups:

```math
\begin{aligned}
\Delta_{\text{median}} = \text{median}(e_{\text{RMS, high quartile}}) - \text{median}(e_{\text{RMS, low quartile}})
\end{aligned}
```

and rank factors by:

```math
\begin{aligned}
\left|\Delta_{\text{median}}\right|
\end{aligned}
```

From the Monte Carlo data, the dominant contributor for both controllers is resonant mode variation (`freq_scale`):

- S curve + standard PD:  $`\Delta_{\text{median}} \approx 7.10`$ arcsec  
- Fourth order + standard PD:  $`\Delta_{\text{median}} \approx 1.12`$ arcsec

This is a very important result. It tells us that post slew pointing performance is most sensitive to uncertainty in modal frequency, not to noise level or disturbance level.

For the S curve case, this sensitivity is severe. The median post slew RMS pointing error shifts from about \(17.09\) arcsec to \(9.99\) arcsec across the resonant mode quartiles. Both values are still above the 7 arcsec requirement, which is consistent with the earlier Monte Carlo finding that S curve violates the pointing requirement in all runs.

For the fourth order case, the same factor shift moves the median from about \(7.05\) arcsec to \(5.93\) arcsec. This lands close to the requirement boundary on the worst side and comfortably below it on the best side, which directly explains why fourth order passes in most runs but still shows a finite number of violations.

The next strongest contributors are damping variation and inertia variation, and both have much smaller impact in the fourth order architecture than in the S curve architecture. This confirms that the fourth order trajectory shaping does not only reduce nominal vibration excitation, it also reduces sensitivity to plant uncertainty.

Finally, disturbance and noise factors are comparatively weak drivers in this post slew window, especially for the fourth order case. That aligns with our closed loop analysis in Section 6.4 and with the frequency sweeps in Section 8.2, where low frequency structural dynamics dominate the final pointing stability outcome.

In short, the histogram based impact study agrees with the rest of the report:

1. the critical robustness driver is modal frequency uncertainty  
2. fourth order shaping keeps the system closer to requirement compliance under that uncertainty  
3. S curve remains structurally more vulnerable in post slew precision pointing

## 9 Conclusion

In this project, we designed and validated a complete attitude control architecture for a 180 degree rest to rest slew of a flexible spacecraft, with the main objective of acheiving post slew pointing stability. The architecture combines feedforward setpoint shaping with feedback PD control, and the full workflow was tested on both linearized models and the nonlinear Basilisk simulation.

The core result is clear. Fourth order setpoint shaping consistently outperforms the S curve baseline in the metrics that matter for this mission. It gives lower post slew RMS pointing error, lower residual vibration, and lower peak control torque. The tradeoff is a slightly higher RMS control torque, which is acceptable in this context because it is exchanged for substantially better structural quieting and precision pointing.

From the closed loop and frequency domain analysis, we confirmed two important mechanisms. First, the feedforward profile is what primarily determines whether structural modes are excited during the slew. Second, the feedback loop mainly governs low frequency tracking and disturbance response, while modal robustness is strongly tied to where trajectory spectral energy sits relative to the flexible resonances and antiresonances.

The Monte Carlo analysis then tested whether these conclusions hold under uncertainty. With 500 randomized trials at plus or minus 20 percent variation, the fourth order architecture remains significantly more robust than S curve. Using the post slew RMS pointing requirement of 7 arcsec over the 60 second post slew window, fourth order shows a finite violation rate, while S curve fails in all runs. This means the fourth order method is not just better in nominal cases, but materially better in our mission conditions.

The parameter impact histograms also identify what drives the remaining risk. Modal frequency uncertainty is the dominant contributor to pointing degradation for both controllers, followed by damping and inertia variations. Disturbance and sensor noise are secondary in the post slew window for this mission setup. This matches the transfer function analysis and closes the loop between theory, simulation, and uncertainty quantification.

So the final engineering takeaway is direct. If the mission objective is precise post slew pointing on a flexible spacecraft, fourth order setpoint shaping with standard PD feedback is the stronger baseline architecture than S curve shaping with the same feedback law for our project. The next improvement path is not changing the basic architecture, but reducing sensitivity to modal frequency uncertainty through better mode identification, more robust shaping against mode drift, or adaptive retuning around the first flexible mode.

## Author
**[Jomin Joseph Karukakalam](https://www.linkedin.com/in/jomin-joseph-karukakalam-601955225)**


