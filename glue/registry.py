"""Central import hub so gin can discover every component we define.

Gin resolves `@Name` references by class name, but a class is only registered
once Python has imported the module that defines it. Upstream achieves this via
`rgfn/__init__.py`, which imports its whole package tree. We mirror that here:
importing `glue` (which imports this module) pulls in every subpackage, so all of
our `@gin.configurable` oracles / rewards / samplers / proxies are visible to gin.

When you add a new module that defines a gin-configurable class, make sure it is
imported here (directly, or via its subpackage `__init__`). If gin raises
"No configurable matching ..." it almost always means the defining module was not
imported on the startup path.
"""

# Importing the subpackages triggers registration of everything they expose.
from glue import active_learning  # noqa: F401
from glue import datasets  # noqa: F401
from glue import fixed_reward  # noqa: F401
from glue import metrics  # noqa: F401
from glue import oracles  # noqa: F401
from glue import proxies  # noqa: F401
from glue import rewards  # noqa: F401
from glue import samplers  # noqa: F401
