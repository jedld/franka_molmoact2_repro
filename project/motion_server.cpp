// sudo apt-get install libcpprest-dev
// g++ -std=c++11  bdm_motion_server.cpp -o bdm_motion_server -lcpprest -lssl -lcrypto -lboost_system
// test:
// curl -X POST http://192.168.2.100:34568 -H "Content-Type: application/json" -d '{"moveToCartesian": [0.5, 0, 0.3]}'
//
// Changes from motion_server.cpp:
//   - Constructor now calls automaticErrorRecovery() on startup to clear any stale error state
//   - moveToCartesian() catches franka::Exception, calls automaticErrorRecovery() to
//     unlock the robot, then returns HTTP 400 with a "collision_recovery:" prefix so
//     the Python side can decide what motion to perform next.
//   - initialize() uses the same goHome() prompt as motion_server (press Enter before moving).

#include <cpprest/http_listener.h>
#include <cpprest/json.h>
#include <iomanip>
#include <mutex>
#include "common.h"

using namespace web;
using namespace web::http;
using namespace web::http::experimental::listener;

class RobotHandler {
    franka::Robot robot;
    franka::Gripper gripper;
    std::mutex robot_mutex;  // libfranka is not thread-safe; serialize all robot calls
public:
    // RealtimeConfig::kIgnore lets robot.control() run on a generic
    // (non-PREEMPT_RT) kernel. Trade-off: libfranka emits warnings on
    // missed 1 ms deadlines and the cubic trajectories may show minor
    // jerks. Chosen because the RT kernel on airscan4 caused desktop
    // freezes requiring hard resets, which lost more trial data than
    // off-realtime jitter ever would, even after rt_install.sh +
    // rt_setup.sh mitigations were applied. See README caveat 4.
    RobotHandler(const std::string& ip_addr)
        : robot(ip_addr, franka::RealtimeConfig::kIgnore), gripper(ip_addr) {
        // Clear any pre-existing error state left over from a previous run
        try {
            robot.automaticErrorRecovery();
        } catch (const franka::Exception& e) {
            // No error to recover from — this is fine
            std::cerr << "Startup error recovery (may be benign): " << e.what() << std::endl;
        }
        // Loosen Franka's collision/reflex thresholds. Under
        // RealtimeConfig::kIgnore the cubic joint and Cartesian trajectories
        // run by motion_server produce slightly noisier torque profiles than
        // the original RT path, which trips Desk's default thresholds
        // (observed 2026-05-23 in Room 210: cartesian_reflex during goHome
        // and goHomeJoints from a moderate non-home start pose, with
        // control_command_success_rate=1 — so not a deadline-miss). The
        // operator keeps a hand on the E-stop during every trial, so Desk's
        // reflex is redundant as a safety layer here -- the human is the
        // floor. Values match libfranka's generate_joint_position_motion
        // example.
        //
        // Note: setDefaultBehavior() is still NOT called -- it would set
        // values that are too tight for the robot's pose-dependent gravity
        // loading. This explicit setCollisionBehavior is the only override.
        // Acceleration thresholds set near Franka's joint torque (87 Nm) and
        // cartesian force (100 N) hardware limits so the reflex effectively
        // ignores transient spikes during the slightly noisy kIgnore motion
        // profile. Nominal thresholds left moderate so steady-state contact
        // is still caught. With the operator's E-stop in hand throughout
        // every trial, this trade-off is safe and was what got homing past
        // the cartesian_reflex/configured-force-thresholds-reached failures
        // observed in Room 210 on 2026-05-23.
        try {
            robot.setCollisionBehavior(
                {{87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0}},
                {{87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0}},
                {{30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0}},
                {{30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0}},
                {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                {{50.0, 50.0, 50.0, 50.0, 50.0, 50.0}},
                {{50.0, 50.0, 50.0, 50.0, 50.0, 50.0}}
            );
        } catch (const franka::Exception& e) {
            std::cerr << "[ctor] setCollisionBehavior failed: " << e.what() << std::endl;
        }
    }

    // Cubic joint-position trajectory from current q to the canonical
    // home pose used by the GUI. Works under RealtimeConfig::kIgnore for
    // the same reason moveToCartesian does -- robot.control() honours the
    // constructor's RT config; goHome() in common.cpp doesn't (it strictly
    // requires RT scheduling), so this is our fallback when goHome throws.
    // 10 s tf (longer than DEFAULT_MOTION_TIME for moveToCartesian) keeps
    // wrist-joint angular velocity low enough that the EE doesn't generate
    // cartesian force spikes during the homing swing.
    void goHomeJoints(double tf = 10.0) {
        const std::array<double, 7> q_home = {
            0.0, -M_PI_4, 0.0, -3.0 * M_PI_4, 0.0, M_PI_2, M_PI_4
        };
        std::array<double, 7> q_initial{};
        double time = 0.0;

        robot.control(
            [&time, &q_initial, q_home, tf]
            (const franka::RobotState& state, franka::Duration period) -> franka::JointPositions {
                time += period.toSec();
                if (time == 0.0) {
                    q_initial = state.q;
                }
                std::array<double, 7> q_target = q_initial;
                double t2 = pow(time, 2);
                double t3 = pow(time, 3);
                double tf2 = pow(tf, 2);
                double tf3 = pow(tf, 3);
                for (int i = 0; i < 7; ++i) {
                    double delta = q_home[i] - q_initial[i];
                    double a2 = 3.0 * delta / tf2;
                    double a3 = -2.0 * delta / tf3;
                    q_target[i] = q_initial[i] + a2 * t2 + a3 * t3;
                }
                franka::JointPositions output = q_target;
                if (time >= tf) {
                    std::cout << "[goHomeJoints] reached home pose" << std::endl;
                    return franka::MotionFinished(output);
                }
                return output;
            }
        );
    }

    void initialize() {
        // Same startup behaviour as motion_server: prompt user before moving.
        // Two-step home strategy: try goHome() (RT-only) first; if that
        // fails because of the kernel, fall back to goHomeJoints() which
        // is a kIgnore-friendly cubic joint trajectory. If both fail, log
        // and continue -- the rest of motion_server (readState,
        // moveToCartesian, gripper) still works, the robot just stays
        // wherever it was at boot.
        bool homed = false;
        try {
            goHome(robot);
            homed = true;
        } catch (const franka::Exception& e) {
            std::cerr << "[initialize] goHome skipped: " << e.what() << std::endl;
        }
        if (!homed) {
            // goHome can leave the robot in Reflex mode (e.g.
            // cartesian_reflex on a force-threshold trip mid-motion).
            // Reflex mode rejects any further robot.control() call, which
            // would make the fallback fail with "command not possible in
            // the current mode (Reflex)". Clear it before retrying.
            try {
                robot.automaticErrorRecovery();
                std::cerr << "[initialize] cleared Reflex state, retrying..." << std::endl;
            } catch (const franka::Exception& e) {
                std::cerr << "[initialize] pre-fallback recovery failed: "
                          << e.what() << std::endl;
            }
            std::cerr << "[initialize] Falling back to goHomeJoints "
                      << "(cubic joint trajectory under kIgnore)..." << std::endl;
            try {
                goHomeJoints();
                homed = true;
            } catch (const franka::Exception& e) {
                std::cerr << "[initialize] goHomeJoints also failed: "
                          << e.what() << std::endl;
                try {
                    robot.automaticErrorRecovery();
                } catch (...) {}
                std::cerr << "[initialize] Continuing without homing. "
                          << "Manually jog the arm via the Desk hand-guidance "
                          << "before sending moveToCartesian." << std::endl;
            }
        }
        try {
            gripper.homing();
        } catch (const franka::Exception& e) {
            std::cerr << "[initialize] gripper.homing skipped: " << e.what() << std::endl;
        }

        franka::Model model = robot.loadModel();
        const franka::RobotState& robot_state = robot.readOnce();
        std::array<double, 16> initial_pose = robot_state.O_T_EE;
        for(int i = 0; i < 16; i++)
        {
            std::cout << initial_pose[i] << ", ";
            if (i % 4 == 3)
            {
                std::cout << std::endl;
            }
        }
        getRotationAngles();
    }


    std::tuple<double, double, double> getRotationAngles(const std::array<double, 16>* custom_pose = nullptr, bool verbose = true) {
        std::array<double, 16> pose;
        if (custom_pose) {
            pose = *custom_pose;
        } else {
            const franka::RobotState& robot_state = robot.readOnce();
            pose = robot_state.O_T_EE;
        }

        double Beta = atan2(-pose[2], sqrt(pow(pose[0], 2) + pow(pose[1], 2)));
        double Alpha = 0.0;
        double Gamma = 0.0;
        double cosBeta = cos(Beta);
        if (abs(cosBeta) < 1e-5) {
            if (Beta > 0){
                Beta = M_PI_2;
                Gamma = atan2(pose[4], pose[5]);
            }else{
                Beta = -M_PI_2;
                Gamma = -atan2(pose[4], pose[5]);
            }
        }else{
            Alpha = atan2(pose[1]/cosBeta, pose[0]/cosBeta);
            Gamma = atan2(pose[6]/cosBeta, pose[10]/cosBeta);
        }

        if (verbose) {
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "Alpha(Z): " << Alpha << ", Beta(Y): " << Beta << ", Gamma(X): " << Gamma << " radians" << std::endl;
            // in degrees
            std::cout << "Alpha(Z): " << Alpha * 180 / M_PI << ", Beta(Y): " << Beta * 180 / M_PI << ", Gamma(X): " << Gamma * 180 / M_PI << " degrees" << std::endl;
            // xyz coordinates
            std::cout << "X: " << pose[12] << ", Y: " << pose[13] << ", Z: " << pose[14] << " meters" <<  std::endl;
        }
        return std::make_tuple(Alpha, Beta, Gamma);
    }


    bool isValidCartesianPose(const std::vector<float>& numbers) {
        const double sphere_radius = 0.855;
        double xf = numbers[0];
        double yf = numbers[1];
        double zf = numbers[2];
        if (pow(xf, 2) + pow(yf, 2) + pow(zf, 2) > pow(sphere_radius, 2)){
            throw std::runtime_error("The desired position is outside the workspace.");
            return false;
        }
        if (zf < 0.015){
            throw std::runtime_error("The desired position is too close to the table.");
            return false;
        }
        return true;
    }

    bool isValidJointPose(const std::array<double, 7>& q) {
        // Franka Emika Panda joint limits (radians)
        static const std::array<double, 7> q_min = {
            -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973
        };
        static const std::array<double, 7> q_max = {
            2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973
        };
        for (int i = 0; i < 7; ++i) {
            if (q[i] < q_min[i] || q[i] > q_max[i]) {
                std::cerr << "Joint " << (i + 1) << " target " << q[i]
                          << " rad outside ["
                          << q_min[i] << ", " << q_max[i] << "]" << std::endl;
                return false;
            }
        }
        return true;
    }

    // moveToJointPose: cubic joint-position trajectory from current q to
    // user-supplied target q. Same kIgnore-friendly control pattern as
    // goHomeJoints, generalized to any 7-DOF joint target. Intended for
    // VLAs that emit absolute joint poses (e.g. MolmoAct2-DROID's
    // control_mode == "absolute joint pose"), where moveToCartesian is
    // not applicable. Accepts at least 7 floats (joint targets in
    // radians); optionally an 8th float for motion time tf (clamped to
    // >= 5 s, same floor as moveToCartesian). Returns the final 7 joint
    // angles read back from the robot post-motion.
    std::vector<double> moveToJointPose(const std::vector<float> &numbers) {
        std::lock_guard<std::mutex> lock(robot_mutex);
        std::vector<double> final_q(7, 0.0);
        if (numbers.size() < 7) {
            throw std::runtime_error(
                "moveToJointPose requires at least 7 joint angles (rad).");
        }
        std::array<double, 7> q_target_raw;
        for (int i = 0; i < 7; ++i) {
            q_target_raw[i] = static_cast<double>(numbers[i]);
        }
        if (!isValidJointPose(q_target_raw)) {
            throw std::runtime_error(
                "Joint target outside Franka joint limits.");
        }
        double tf = 5.0;
        if (numbers.size() >= 8) {
            tf = static_cast<double>(numbers[7]);
            if (tf < 5.0) {
                tf = 5.0;
            }
        }
        std::array<double, 7> q_initial{};
        double time = 0.0;
        const std::array<double, 7> q_target = q_target_raw;

        try {
            robot.control(
                [&time, &q_initial, q_target, tf]
                (const franka::RobotState& state, franka::Duration period)
                    -> franka::JointPositions {
                    time += period.toSec();
                    if (time == 0.0) {
                        q_initial = state.q;
                    }
                    std::array<double, 7> q_cur = q_initial;
                    double t2 = pow(time, 2);
                    double t3 = pow(time, 3);
                    double tf2 = pow(tf, 2);
                    double tf3 = pow(tf, 3);
                    for (int i = 0; i < 7; ++i) {
                        double delta = q_target[i] - q_initial[i];
                        double a2 = 3.0 * delta / tf2;
                        double a3 = -2.0 * delta / tf3;
                        q_cur[i] = q_initial[i] + a2 * t2 + a3 * t3;
                    }
                    franka::JointPositions output = q_cur;
                    if (time >= tf) {
                        std::cout << time
                                  << "sec : End of joint motion ........."
                                  << std::endl;
                        return franka::MotionFinished(output);
                    }
                    return output;
                }
            );
        } catch (const franka::Exception &ex) {
            std::cerr << "franka::Exception during moveToJointPose: "
                      << ex.what() << std::endl;
            try {
                robot.automaticErrorRecovery();
            } catch (const franka::Exception &re) {
                throw std::runtime_error(
                    std::string("collision_recovery_failed: original=")
                    + ex.what() + " recovery=" + re.what());
            }
            throw std::runtime_error(
                std::string("collision_recovery: ") + ex.what());
        }

        franka::RobotState st = robot.readOnce();
        for (int i = 0; i < 7; ++i) {
            final_q[i] = st.q[i];
        }
        return final_q;
    }

    // moveToCartesian accepts three float numbers in a vector
    std::vector<double> moveToCartesian(const std::vector<float> &numbers)
    {
        std::lock_guard<std::mutex> lock(robot_mutex);
        std::vector<double> final_coords(6, 0.0);
        if (numbers.size() < 3)
        {
            throw std::runtime_error("Please provide at least 3 float numbers (x,y,z,t).");
            return final_coords;
        }
        if (!isValidCartesianPose(numbers))
        {
            return final_coords;
        }
        double xf = numbers[0];
        double yf = numbers[1];
        double zf = numbers[2];
        std::array<double, 16> initial_pose;
        std::array<double, 16> final_pose;
        double time = 0.0;
        double tf = 0.0;
        double Alpha = 0.0, Beta = 0.0, Gamma = 0.0l;
        double deltaAlpha = 0.0, deltaBeta = 0.0, deltaGamma = 0.0;
        double Alphaf = 0.0, Betaf = 0.0, Gammaf = 0.0;
        bool is_rotation = false;
        const double DEFAULT_MOTION_TIME = 5.0;

        if (numbers.size() < 4)
        {
            tf = DEFAULT_MOTION_TIME;
        }
        else if (numbers.size() >= 4)
        {
            tf = numbers[3];
            if (tf < DEFAULT_MOTION_TIME)
            {
            tf = DEFAULT_MOTION_TIME;
            }
        }

        if (numbers.size() >= 5)
        {
            deltaAlpha = numbers[4];
            // must be between -90 to 90 degrees
            if (deltaAlpha < -90 || deltaAlpha > 90)
            {
                deltaAlpha = 0.0;
                std::cerr << "Error: deltaAlpha must be between -90 and 90 degrees." << std::endl;
            }
            deltaAlpha = deltaAlpha * M_PI / 180;  // in radians
            is_rotation = true;
        }

        if (numbers.size() >= 6)
        {
            deltaBeta = numbers[5];
            // must be between -90 and 90 degrees
            if (deltaBeta < -90 || deltaBeta > 90)
            {
                deltaBeta = 0.0;
                std::cerr << "Error: deltaBeta must be between -90 and 90 degrees." << std::endl;
            }
            deltaBeta = deltaBeta * M_PI / 180;  // in radians
        }
        if (numbers.size() >= 7)
        {
            deltaGamma = numbers[6];
            // must be between -90 and 90 degrees
            if (deltaGamma < -90 || deltaGamma > 90)
            {
                deltaGamma = 0.0;
                std::cerr << "Error: deltaGamma must be between -90 and 90 degrees." << std::endl;
            }
            deltaGamma = deltaGamma * M_PI / 180;  // in radians
        }

        try
        {
            auto trajectory_callback = [this, xf, yf, zf, tf, is_rotation, deltaAlpha, deltaBeta, deltaGamma,
                                        &time, &Alpha, &Beta, &Gamma, &Alphaf, &Betaf, &Gammaf, &initial_pose, &final_pose](
                                           const franka::RobotState &robot_state,
                                           franka::Duration period) -> franka::CartesianPose
            {
                time += period.toSec();

                if (time == 0.0)
                {
                    // Read the initial pose to start the motion from in the first time step.
                    initial_pose = robot_state.O_T_EE;
                    for(int i = 0; i < 16; i++)
                    {
                        std::cout << initial_pose[i] << ", ";
                        if (i % 4 == 3)
                        {
                            std::cout << std::endl;
                        }
                    }
                    if (is_rotation){
                        std::tuple<double, double, double> angles = getRotationAngles(&initial_pose, false);
                        Alpha = std::get<0>(angles);
                        Beta = std::get<1>(angles);
                        Gamma = std::get<2>(angles);

                        Alphaf = Alpha + deltaAlpha;
                        Betaf = Beta + deltaBeta;
                        Gammaf = Gamma + deltaGamma;
                    }
                }

                // cubic polynomial trajectory
                franka::CartesianPose pose_desired = initial_pose;
                double t2 = pow(time, 2);
                double t3 = pow(time, 3);
                double tf2 = pow(tf, 2);
                double tf3 = pow(tf, 3);

                double x0 = pose_desired.O_T_EE[12];
                double a2 = 3 * (xf - x0) / tf2;
                double a3 = -2 * (xf - x0) / tf3;
                double xt = x0 + a2 * t2 + a3 * t3;
                double vx = 2 * a2 * time + 3 * a3 * t2;

                double y0 = pose_desired.O_T_EE[13];
                a2 = 3 * (yf - y0) / tf2;
                a3 = -2 * (yf - y0) / tf3;
                double yt = y0 + a2 * t2 + a3 * t3;
                double vy = 2 * a2 * time + 3 * a3 * t2;

                double z0 = pose_desired.O_T_EE[14];
                a2 = 3 * (zf - z0) / tf2;
                a3 = -2 * (zf - z0) / tf3;
                double zt = z0 + a2 * t2 + a3 * t3;
                double vz = 2 * a2 * time + 3 * a3 * t2;
                bool stop = fabs(vx) < 0.0001 && fabs(vy) < 0.0001 && fabs(vz) < 0.0001 && time > 1.0;

                pose_desired.O_T_EE[12] = xt;
                pose_desired.O_T_EE[13] = yt;
                pose_desired.O_T_EE[14] = zt;

                if(is_rotation){
                    a2 = 3 * (Alphaf - Alpha) / tf2;
                    a3 = -2 * (Alphaf - Alpha) / tf3;
                    double Alphat = Alpha + a2 * t2 + a3 * t3;
                    double vAlpha = 2 * a2 * time + 3 * a3 * t2;

                    a2 = 3 * (Betaf - Beta) / tf2;
                    a3 = -2 * (Betaf - Beta) / tf3;
                    double Betat = Beta + a2 * t2 + a3 * t3;
                    double vBeta = 2 * a2 * time + 3 * a3 * t2;

                    a2 = 3 * (Gammaf - Gamma) / tf2;
                    a3 = -2 * (Gammaf - Gamma) / tf3;
                    double Gammat = Gamma + a2 * t2 + a3 * t3;
                    double vGamma = 2 * a2 * time + 3 * a3 * t2;

                    // Rotation matrix (ZYX Euler: Alpha=Z, Beta=Y, Gamma=X)
                    pose_desired.O_T_EE[0] = cos(Alphat) * cos(Betat);
                    pose_desired.O_T_EE[1] = sin(Alphat) * cos(Betat);
                    pose_desired.O_T_EE[2] = -sin(Betat);
                    pose_desired.O_T_EE[4] = cos(Alphat) * sin(Betat) * sin(Gammat) - sin(Alphat) * cos(Gammat);
                    pose_desired.O_T_EE[5] = sin(Alphat) * sin(Betat) * sin(Gammat) + cos(Alphat) * cos(Gammat);
                    pose_desired.O_T_EE[6] = cos(Betat) * sin(Gammat);
                    pose_desired.O_T_EE[8] = cos(Alphat) * sin(Betat) * cos(Gammat) + sin(Alphat) * sin(Gammat);
                    pose_desired.O_T_EE[9] = sin(Alphat) * sin(Betat) * cos(Gammat) - cos(Alphat) * sin(Gammat);
                    pose_desired.O_T_EE[10] = cos(Betat) * cos(Gammat);
                    stop = stop && fabs(vAlpha) < 0.0001 && fabs(vBeta) < 0.0001 && fabs(vGamma) < 0.0001;
                }

                if (time >= tf || stop)
                {
                    std::cout << std::endl << time << "sec : End of motion ............." << std::endl;
                    final_pose = pose_desired.O_T_EE;
                    return franka::MotionFinished(pose_desired);
                }

                return pose_desired;
            };

            robot.control(trajectory_callback);
        }
        catch (const franka::Exception &ex)
        {
            std::cerr << "franka::Exception during motion: " << ex.what() << std::endl;

            // Unlock the robot so it can accept new commands.
            // The Python side is responsible for any recovery motion after this.
            try {
                std::cerr << "Attempting automatic error recovery..." << std::endl;
                robot.automaticErrorRecovery();
                std::cerr << "Robot unlocked. Returning 400 to caller." << std::endl;
            } catch (const franka::Exception& recovery_ex) {
                std::cerr << "Error recovery failed: " << recovery_ex.what() << std::endl;
                throw std::runtime_error(
                    std::string("collision_recovery_failed: original=") + ex.what() +
                    " recovery=" + recovery_ex.what());
            }

            // The "collision_recovery:" prefix lets mainutilsedgrasper.py detect this case.
            throw std::runtime_error(std::string("collision_recovery: ") + ex.what());
        }

        final_pose = robot.readOnce().O_T_EE;
        // Pass the already-read pose to avoid a redundant readOnce inside
        // getRotationAngles. The actual segfault that motivated the
        // closeGripper / openGripper empty-vector fixes (ASan, 2026-05-05)
        // turned out to be in the gripper path, not here, but this is still
        // a small efficiency win.
        std::tuple<double, double, double> final_angles = getRotationAngles(&final_pose);
        final_coords[0] = final_pose[12];
        final_coords[1] = final_pose[13];
        final_coords[2] = final_pose[14];
        final_coords[3] = std::get<0>(final_angles) * 180.0 / M_PI;  // Alpha (Z) in degrees
        final_coords[4] = std::get<1>(final_angles) * 180.0 / M_PI;  // Beta (Y) in degrees
        final_coords[5] = std::get<2>(final_angles) * 180.0 / M_PI;  // Gamma (X) in degrees
        return final_coords;
    }

    std::string closeGripper(const std::vector<float> &numbers) {
        std::lock_guard<std::mutex> lock(robot_mutex);
        try {
            // The GUI calls closeGripper with no payload (`{"closeGripper": []}`),
            // which means numbers can be empty. Accessing numbers[0] on an empty
            // vector is undefined behavior and was the actual cause of the
            // segfaults that looked like they came from moveToCartesian
            // (ASan trace 2026-05-05 confirmed). Default to a small 1 cm target
            // width so the fingers SQUEEZE small objects (a wide 4 cm target
            // left the fingers apart on a ~2-3 cm toy carrot).
            double grasping_width = numbers.empty() ? 0.01 : static_cast<double>(numbers[0]);
            franka::GripperState gripper_state = gripper.readOnce();
            if (gripper_state.max_width < grasping_width) {
                return "Object is too large for the current fingers on the gripper: " + std::to_string(gripper_state.max_width);
            }
            // grasp(width, speed, force, epsilon_inner, epsilon_outer). FORCE was
            // 0.0 N -> the fingers reached the target width but applied NO
            // clamping force, so a light, round toy carrot was only touched
            // (and rolled away) rather than held (observed 2026-05-25). Use
            // 40 N and a wide epsilon so the fingers contact the object and
            // clamp it firmly whatever its exact width (0.5-6 cm "succeeds").
            if (!gripper.grasp(grasping_width, 0.1, 40.0, 0.05, 0.05)) {
                return "Failed to grasp object.";
            }
            std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(100));
            gripper_state = gripper.readOnce();
            if (!gripper_state.is_grasped) {
                return "Object lost.";
            }
        } catch (franka::Exception const& e) {
            return e.what();
        }
        return "Object grasped successfully.";
    }

    std::string openGripper(const std::vector<float> &numbers)
    {
        std::lock_guard<std::mutex> lock(robot_mutex);
        try
        {
            franka::GripperState gripper_state = gripper.readOnce();
            // Same empty-vector guard as closeGripper: GUI sends
            // {"openGripper": []} with no width or speed argument.
            double speed = numbers.empty() ? 0.1 : static_cast<double>(numbers[0]);
            std::cout << "Grasped object, will release it now." << std::endl;
            gripper.move(gripper_state.max_width, speed);
            gripper.stop();
        }
        catch (franka::Exception const &e)
        {
            return e.what();
        }
        return "Gripper opened successfully.";
    }

    // Returns current end-effector pose: [x, y, z, alpha_deg, beta_deg, gamma_deg].
    // Used by Python to obtain the actual current rotation before issuing a reset move,
    // since after a collision the arm may have stopped at an unknown intermediate angle.
    std::vector<double> readState()
    {
        std::lock_guard<std::mutex> lock(robot_mutex);
        const franka::RobotState& state = robot.readOnce();
        std::array<double, 16> pose = state.O_T_EE;
        auto [alpha, beta, gamma] = getRotationAngles(&pose, false);
        return {pose[12], pose[13], pose[14],
                alpha * 180.0 / M_PI,
                beta  * 180.0 / M_PI,
                gamma * 180.0 / M_PI};
    }

    // Returns current joint angles [q1..q7] in radians. Companion to
    // readState for joint-space VLAs (MolmoAct2-DROID) that need the
    // actual robot q as state input rather than the Cartesian pose.
    std::vector<double> readJointState()
    {
        std::lock_guard<std::mutex> lock(robot_mutex);
        const franka::RobotState& state = robot.readOnce();
        std::vector<double> q(7);
        for (int i = 0; i < 7; ++i) {
            q[i] = state.q[i];
        }
        return q;
    }
};

class RobotRestAPI {
public:
    RobotRestAPI(utility::string_t url, const std::string& robot_ip)
        : m_listener(url), robotHandler(robot_ip) {
        m_listener.support(methods::POST, std::bind(&RobotRestAPI::handle_post, this, std::placeholders::_1));
        robotHandler.initialize();
    }

    pplx::task<void> open() { return m_listener.open(); }
    pplx::task<void> close() { return m_listener.close(); }

private:
    void handle_post(http_request request) {
        request
            .extract_json()
            .then([](json::value body)
                  {
                      std::map<std::string, std::vector<float>> data;
                      for (const auto& item : body.as_object())
                      {
                          if (!item.second.is_array())
                          {
                              throw std::runtime_error("All values must be arrays.");
                          }
                          std::vector<float> numbers;
                          for (const auto& num : item.second.as_array())
                          {
                              if (!num.is_number())
                              {
                                  throw std::runtime_error("All array elements must be numbers.");
                              }
                              numbers.push_back(static_cast<float>(num.as_double()));
                          }
                          data[item.first] = numbers;
                      }
                      return data;
                  })
            .then([this](std::map<std::string, std::vector<float>> data)
                  {
                json::value response;
                for (const auto& item : data)
                {
                    const std::string& key = item.first;
                    const std::vector<float>& numbers = item.second;

                    if (key == "moveToCartesian") {
                        std::vector<double> final_coords = robotHandler.moveToCartesian(numbers);
                        for (int i = 0; i < 6; ++i) {
                            response[key][i] = json::value::number(final_coords[i]);
                        }
                    } else if (key == "moveToJointPose") {
                        std::vector<double> final_q = robotHandler.moveToJointPose(numbers);
                        for (int i = 0; i < 7; ++i) {
                            response[key][i] = json::value::number(final_q[i]);
                        }
                    } else if (key == "closeGripper") {
                        std::string message = robotHandler.closeGripper(numbers);
                        response[key] = json::value::string(message);
                    } else if (key == "openGripper") {
                        std::string message = robotHandler.openGripper(numbers);
                        response[key] = json::value::string(message);
                    } else if (key == "readState") {
                        std::vector<double> state = robotHandler.readState();
                        for (int i = 0; i < 6; ++i) {
                            response[key][i] = json::value::number(state[i]);
                        }
                    } else if (key == "readJointState") {
                        std::vector<double> q = robotHandler.readJointState();
                        for (int i = 0; i < 7; ++i) {
                            response[key][i] = json::value::number(q[i]);
                        }
                    } else {
                        std::cout << "Invalid command: " << key << std::endl;
                        response["Response"] = json::value::string("Invalid command.");
                    }
                }
                return response; })
            .then([=](json::value response)
                  { request.reply(status_codes::OK, response); })
            .then([=](pplx::task<void> t)
                  {
                try {
                    t.get();
                }
                catch (const std::exception &e) {
                    request.reply(status_codes::BadRequest, json::value::string(e.what()));
                } });
    }

    http_listener m_listener;
    RobotHandler robotHandler;
};


#include <iostream>

int main(int argc, char* argv[]) {
    std::string port = "34568";
    std::string address = "http://0.0.0.0:" + port;
    utility::string_t utility_address = utility::conversions::to_string_t(address);
    std::string robot_ip = "192.168.2.100";

    if (argc == 2) {
        robot_ip = argv[1];
    }

    RobotRestAPI api(utility_address, robot_ip);

    try {
        api.open().wait();
        std::cout << "Listening at: " << address << std::endl;
        std::cout << "Press ENTER to exit." << std::endl;
        std::string line;
        std::getline(std::cin, line);
        api.close().wait();
    }
    catch (std::exception const & e) {
        std::cout << e.what() << std::endl;
    }

    return 0;
}
