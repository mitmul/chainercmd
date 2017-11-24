from chainer.dataset import convert
from chainer.training import extension
from chainer.training.extensions import evaluator


def evaluator_creator(iterator, target, devices, **kwargs):
    """A sample evaluator creator.

    An evaluator creator method should return an evaluator extension object.
    Once an evaluator creator method is specified in the config YAML, the
    method will take iterator object on ``valid`` dataset, and an optimizer
    object, device dictionary, and "args" dictionary defined in the config
    YAML. You can make a custom evaluator with those objects, and then please
    return it from this function.

    """
    return evaluator.Evaluator(
        iterator, target, converter=convert.concat_examples,
        device=devices['main'], eval_hook=None, eval_func=None)

