# Latent Failure

Discovering and Exploiting Failure Manifolds in Autonomous Robotics

Latent Failure is a research framework for systematically discovering, modeling, and exploiting failure modes in autonomous robotic systems. Unlike conventional approaches that rely on random domain variations or handcrafted stress tests, this project focuses on learning latent failure manifolds. These are structured regions of the state, observation, and dynamics space where autonomous policies exhibit brittle behavior. By identifying these regions, it is possible to reveal weaknesses in perception, control, and recovery pipelines that are difficult to detect through standard testing.

The failure scenarios identified by the framework are used to evaluate policy robustness, train recovery strategies, and improve overall system resilience. The framework is developed with simulation-first robotics in mind, using tools such as Isaac Sim and ROS 2. It targets safety-critical tasks including target tracking, navigation, and autonomous decision-making under uncertainty, where undetected failures can have significant consequences.

The primary objective of the project is to uncover high-impact, structured failure scenarios using generative and adversarial methods. These learned scenarios allow autonomous policies to be stress-tested beyond naive randomization, exposing blind spots and fragile behaviors. In addition, failure modes are analyzed and clustered in latent space to improve interpretability and provide insight into the mechanisms underlying system failures. Finally, the framework leverages these insights to train recovery behaviors and robustness mechanisms, with the goal of producing more resilient autonomous systems.

Modern autonomous systems can perform well under nominal conditions while remaining vulnerable to rare or structured edge cases. Latent Failure frames robustness testing as a learning problem, enabling researchers to systematically identify and address the weakest behaviors before deployment. This framework serves as a testbed for research in failure-aware autonomy, resilient control, and safety-critical robotics.

The latent failure project is built on three models: 
1. World Model => its job: "If I take action uT in state xT, what happens next?"
2. Failure Predictor => its job: "If I continue like this, will I be able to successfully complete the task?"
3. Policy Model => its job: hesistate when uncertain and fall back to different decision scenarios created prior to diving into the required task (I.e. I can do this task X different ways), which similar to human task execution:
Start Task -> Failing -> Reiterate -> New Solution -> Goal Achieved
