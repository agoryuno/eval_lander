# eval_lander

This provides the `EvalLander` class, which inherits from SB3's standard `LunarLander` while overriding its `reset()` method.

The purpose of this is to facilitate direct comparisons between `LunarLander` models by evaluating them with a stable set of initial conditions.
Needed to note that by "initital conditions" we mean only the starting position, rotation, and momentum of the lander - differences in landscapes
are not affected.

The standard `LunarLander` class objects reset to random initial conditions. That means that SB3's `evaluate_policy()` function essentially 
feeds white noise through the evaluated model and produces significantly different mean and std of rewards on subsequent evaluation runs.

While the default behavior is true to the purpose of training a model to optimize reward in the space of random initial conditions, it 
makes direct comparisons between models very difficult. Since each run of `evaluate_policy()` produces a random sample of results, you would need either
a very large sample or a very large number of smaller samples to get an idea of how two different models perform. This is especially true for models
which are close in performance, because for such pairs of models even small differences in distributions of results matter greatly.

The modified class `EvalLander` allows you to feed a stable set of initial conditions through a model as many time as required, to compare as many models
as needed, returning results which are determined by the model's own performance and differences in landscape.

## Using the module

To use the `EvalLander` class you need to instantiate it with either the number of episodes you want to run the evaluation for or
an list of tuples, each containing a pair of values for the X and Y coordinates of the force vector to be applied to the lander
at the start of each episode. Before passing the object to `evaluate_policy()` you need to call the object's `reinit()` method to reset
the force vectors:

    from eval_lander import EvalLander

    # Create an environment for 100 episodes
    eval_env = EvalLander(100)
    eval_env.reinit()

    # Pass eval_env.episodes_length as n_eval_episodes argument's value
    mean_reward, std_reward = evaluate_policy(model, eval_env, 
                                              n_eval_episodes=eval_env.episodes_length, 
                                              deterministic=True)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
