from xvla_wlr.agent import *
xvla_agent = XVLAAgent(XVLAAgent.Config.sample())
for i in range(3):
    xvla_agent.learn(
        XVLAObservation.sample(),
        XVLAAction.sample(),
    )
    print("saving")
    xvla_agent.save("/tmp/some-checkpoint-a/checkpoint.json", force=True)
