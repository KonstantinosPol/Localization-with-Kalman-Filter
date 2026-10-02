// ============================================================
//  Sensor Fusion with a Kalman Filter (1D)
//
//  THE PROBLEM THIS SOLVES
//  -----------------------
//  A robot wants to know where it is. It has two sensors:
//
//    1. An accelerometer (IMU). Fast (100 Hz) but noisy. If you
//       just add up its readings to get position, the small errors
//       pile up and your estimate slowly slides away from reality.
//       This is called DRIFT, and it is the central problem in
//       robot localization.
//
//    2. A position sensor (think GPS, or a camera recognising a
//       known landmark). Slow (5 Hz) and noisy, but it does NOT
//       drift - it is always roughly right.
//
//  Neither sensor alone is good enough. A Kalman filter combines
//  them: the IMU fills in the gaps between position fixes, and each
//  position fix pulls the estimate back before drift builds up.
//
//  This program simulates a robot, runs both approaches, and prints
//  how much error each one has. Everything is 1D (motion along a
//  line) so the maths stays small enough to read.
//
//  Build:  g++ -std=c++17 -Wall -o kalman kalman_localization.cpp
//  Run:    ./kalman           (on Windows: kalman.exe)
// ============================================================

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <random>
#include <vector>

// ------------------------------------------------------------
// SIMULATION SETTINGS
// ------------------------------------------------------------
const double DT            = 0.01;   // seconds between IMU readings (100 Hz)
const double DURATION      = 20.0;   // how long to simulate, in seconds
const int    FIX_EVERY     = 20;     // a position fix every 20 steps (5 Hz)

const double ACCEL_NOISE   = 0.30;   // accelerometer noise, m/s^2
const double ACCEL_BIAS    = 0.05;   // small constant offset - the drift culprit
const double POS_NOISE     = 0.50;   // position sensor noise, metres

// ------------------------------------------------------------
// THE TRUE MOTION
//
// The robot accelerates back and forth in a smooth wave. We know
// these exact formulas only because WE are the simulator - the
// filter never sees them. They are our answer key.
// ------------------------------------------------------------
double trueAcceleration(double t) { return  1.5 * std::sin(0.5 * t); }
double trueVelocity(double t)     { return -3.0 * std::cos(0.5 * t) + 3.0; }
double truePosition(double t)     { return -6.0 * std::sin(0.5 * t) + 3.0 * t; }

// ------------------------------------------------------------
// THE FILTER'S STATE
//
// The filter tracks two numbers it cannot measure directly:
// where the robot is, and how fast it is going.
//
// It also tracks how UNSURE it is, in a 2x2 "covariance" matrix P.
// Think of P as the filter's own error bars:
//   P00 = uncertainty in position
//   P11 = uncertainty in velocity
//   P01, P10 = how the two errors are linked (they always are:
//              a wrong velocity makes position wrong next step)
//
// P is symmetric, so P01 and P10 stay equal.
// ------------------------------------------------------------
struct KalmanFilter {
    double position = 0.0;
    double velocity = 0.0;

    double P00 = 1.0,  P01 = 0.0;
    double P10 = 0.0,  P11 = 1.0;

    // --------------------------------------------------------
    // STEP 1: PREDICT
    //
    // "Given the acceleration I just felt, where should I be now?"
    //
    // Basic physics, the same formulas from school:
    //     position += velocity * dt + 0.5 * a * dt^2
    //     velocity += a * dt
    //
    // Then we grow P, because a prediction is always a guess and
    // guesses make us less certain. The formula is P = F*P*F' + Q:
    //     F  describes how the state moves forward
    //     Q  is how much new noise the accelerometer injected
    //
    // The lines below are that matrix product written out by hand
    // for the 2x2 case, so no linear algebra library is needed.
    // --------------------------------------------------------
    void predict(double measuredAccel, double dt, double accelNoise) {
        // --- move the state forward ---
        position += velocity * dt + 0.5 * measuredAccel * dt * dt;
        velocity += measuredAccel * dt;

        // --- F * P * F', with F = [[1, dt], [0, 1]] ---
        double newP00 = P00 + dt * (P10 + P01) + dt * dt * P11;
        double newP01 = P01 + dt * P11;
        double newP10 = P10 + dt * P11;
        double newP11 = P11;

        // --- + Q, the noise this step added ---
        // A noisy acceleration 'a' shifts position by 0.5*a*dt^2
        // and velocity by a*dt, so those are the terms we square.
        double gPos = 0.5 * dt * dt;
        double gVel = dt;
        double var  = accelNoise * accelNoise;

        P00 = newP00 + gPos * gPos * var;
        P01 = newP01 + gPos * gVel * var;
        P10 = newP10 + gVel * gPos * var;
        P11 = newP11 + gVel * gVel * var;
    }

    // --------------------------------------------------------
    // STEP 2: UPDATE (also called "correct")
    //
    // "A position sensor just spoke. How much should I believe it?"
    //
    // The Kalman gain K is the answer, and it is just a ratio:
    //
    //     K = (my uncertainty) / (my uncertainty + sensor noise)
    //
    // If the filter is very unsure, K is close to 1 and it mostly
    // adopts the measurement. If the filter is confident, K is close
    // to 0 and it mostly ignores it. Nobody tunes K by hand - it
    // falls out of the numbers automatically, every single step.
    // That self-tuning is the whole point of a Kalman filter.
    // --------------------------------------------------------
    void update(double measuredPosition, double posNoise) {
        // How wrong was my prediction? ("innovation")
        double innovation = measuredPosition - position;

        // Total expected spread of that error.
        double S = P00 + posNoise * posNoise;

        // Gain for each state. Note velocity gets corrected too,
        // via P10 - even though the sensor never measured velocity.
        // That is the covariance doing its job.
        double K_pos = P00 / S;
        double K_vel = P10 / S;

        position += K_pos * innovation;
        velocity += K_vel * innovation;

        // Shrink the uncertainty: we just learned something.
        // This is P = (I - K*H) * P with H = [1, 0].
        double newP00 = (1.0 - K_pos) * P00;
        double newP01 = (1.0 - K_pos) * P01;
        double newP10 = P10 - K_vel * P00;
        double newP11 = P11 - K_vel * P01;

        P00 = newP00;  P01 = newP01;
        P10 = newP10;  P11 = newP11;
    }
};

// ------------------------------------------------------------
// THE NAIVE APPROACH, for comparison.
//
// Just add up the accelerometer readings and hope. This is called
// dead reckoning. It works fine for a second or two, then drifts.
// ------------------------------------------------------------
struct DeadReckoning {
    double position = 0.0;
    double velocity = 0.0;

    void step(double measuredAccel, double dt) {
        position += velocity * dt + 0.5 * measuredAccel * dt * dt;
        velocity += measuredAccel * dt;
    }
};

int main() {
    // A fixed seed means the same "random" noise every run, so
    // results are reproducible while you experiment.
    std::mt19937 rng(42);
    std::normal_distribution<double> accelNoiseGen(0.0, ACCEL_NOISE);
    std::normal_distribution<double> posNoiseGen(0.0, POS_NOISE);

    KalmanFilter  kf;
    DeadReckoning dr;

    // Both start out knowing exactly where they are.
    kf.position = truePosition(0.0);
    kf.velocity = trueVelocity(0.0);
    dr.position = truePosition(0.0);
    dr.velocity = trueVelocity(0.0);

    // Running totals so we can compute RMS error at the end.
    double sumSqKalman = 0.0;
    double sumSqDead   = 0.0;
    int    sampleCount = 0;

    std::ofstream csv("trajectory.csv");
    csv << "time,truth,dead_reckoning,kalman,uncertainty\n";

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n  time      truth     imu-only     kalman   kf error\n";
    std::cout << "  ------------------------------------------------\n";

    const int totalSteps = static_cast<int>(DURATION / DT);

    for (int step = 0; step < totalSteps; ++step) {
        double t = step * DT;

        // ---- Sensor 1: the IMU, read every single step ----
        // Real accelerometers have a bias: a small constant offset
        // that never averages away. It is what makes drift grow
        // and grow instead of wobbling around zero.
        double measuredAccel = trueAcceleration(t) + ACCEL_BIAS + accelNoiseGen(rng);

        dr.step(measuredAccel, DT);
        kf.predict(measuredAccel, DT, ACCEL_NOISE);

        // ---- Sensor 2: the position fix, only now and then ----
        if (step % FIX_EVERY == 0) {
            double measuredPos = truePosition(t) + posNoiseGen(rng);
            kf.update(measuredPos, POS_NOISE);
        }

        // ---- Score both approaches against the answer key ----
        double truth      = truePosition(t + DT);
        double errKalman  = kf.position - truth;
        double errDead    = dr.position - truth;

        sumSqKalman += errKalman * errKalman;
        sumSqDead   += errDead   * errDead;
        ++sampleCount;

        csv << t << "," << truth << "," << dr.position << ","
            << kf.position << "," << std::sqrt(kf.P00) << "\n";

        // Print a row every 2 seconds so the output stays readable.
        if (step % 200 == 0) {
            std::cout << std::setw(6) << t
                      << std::setw(11) << truth
                      << std::setw(13) << dr.position
                      << std::setw(11) << kf.position
                      << std::setw(11) << errKalman << "\n";
        }
    }

    double rmsKalman = std::sqrt(sumSqKalman / sampleCount);
    double rmsDead   = std::sqrt(sumSqDead   / sampleCount);

    std::cout << "\n  RMS error, IMU only ....... " << rmsDead   << " m\n";
    std::cout << "  RMS error, Kalman filter .. " << rmsKalman << " m\n";
    std::cout << "  Improvement ............... "
              << (rmsDead / rmsKalman) << "x better\n";
    std::cout << "\n  Wrote trajectory.csv - open it in a spreadsheet\n"
              << "  and plot the columns to see the drift visually.\n\n";

    return 0;
}

