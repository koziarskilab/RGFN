from rgfn import ReactionDataFactory

from .fixtures import *  # noqa


def test__data_factory_anchored_reactions(rgfn_data_factory: ReactionDataFactory):
    anchored_reactions = rgfn_data_factory.get_anchored_reactions()
    anchored_disconnections = rgfn_data_factory.get_anchored_disconnections()
    for anchored_reaction, disconnection in zip(anchored_reactions, anchored_disconnections):
        assert anchored_reaction.reversed() == disconnection
        assert anchored_reaction == disconnection.reversed()
