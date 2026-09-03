"""Import the perception modules on a host with no ROS, no CUDA and no OpenCV.

A meta-path finder rather than a list of module names: every attempt to enumerate the
submodules of a stubbed dependency went stale within one import, and the failure reads as
a broken test instead of a missing stub. Anything under a stubbed top-level package is
synthesised on demand, and any attribute of it is a placeholder that is callable,
attributable, iterable and indexable -- modules unpack constants at import time, so a
placeholder that is not iterable fails at import rather than at use.

Deliberately NOT stubbed: numpy, and everything in perception_module itself. The point is
to exercise the real code.
"""
import pathlib
import importlib.abc
import importlib.machinery
import sys
import types

STUBBED = ("rclpy", "cv2", "cv_bridge", "torch", "tf2_ros", "webcolors", "gensim",
           "sentence_transformers", "matplotlib", "PIL", "open3d", "shapely", "scipy",
           "sklearn", "visualization_msgs", "sensor_msgs", "sensor_msgs_py",
           "geometry_msgs", "std_msgs", "nav_msgs", "builtin_interfaces", "lost3dsg",
           "onnxruntime", "transformers", "supervision", "ultralytics", "torchvision",
           "efficientvit", "segment_anything", "groundingdino", "tf_transformations",
           "message_filters", "rosidl_runtime_py", "ament_index_python")


class _AnyMeta(type):
    """Class-level attribute access must also yield a stub: enums are read off the CLASS
    (`LoggingSeverity.DEBUG`), and `__getattr__` on the instance never sees that."""
    def __getattr__(cls, name):
        v = Any()
        setattr(cls, name, v)
        return v


class Any(metaclass=_AnyMeta):
    def __init__(self, *a, **k): pass
    def __getattr__(self, n): return Any()
    def __call__(self, *a, **k): return Any()
    def __iter__(self): return iter(())
    def __getitem__(self, k): return Any()
    def __len__(self): return 0
    def __bool__(self): return False
    def __float__(self): return 0.0
    def __int__(self): return 0
    def __repr__(self): return "<stub>"


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        # Capitalised names are used as base classes and constructors, so they must be
        # types; everything else can be an instance.
        v = type(name, (Any,), {}) if name[:1].isupper() else Any()
        setattr(self, name, v)
        return v


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in STUBBED:
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        module.__path__ = []          # let it act as a package


def _real_msg_types():
    """Build lost3dsg.msg types with the REAL field sets, read from the .msg files.

    A generated ROS message has __slots__ and raises AttributeError when you set a field
    it does not declare. A stub that accepts any attribute is not a stub of that -- it is
    a stub of something more permissive, and it hides exactly the defect that cost this
    project eight runs: a blind setattr of a field the message does not have.
    """
    # Derived from the MODULE UNDER TEST, not from this file: a copy of this stub run
    # beside a copy of the tree used to look for msg/ next to ITSELF, not find it, and
    # silently fall back to the permissive stub -- which accepts any attribute, so the
    # very defect this exists to catch passed. A missing msg/ is now an error, because a
    # check that quietly becomes weaker is worse than one that is absent.
    here = pathlib.Path(sys.path[0]).resolve() if sys.path and sys.path[0] else pathlib.Path.cwd()
    for base in (here, *here.parents):
        msg_dir = base / "msg"
        if msg_dir.is_dir() and any(msg_dir.glob("*.msg")):
            break
    else:
        raise RuntimeError(
            f"no msg/ directory with .msg files found at or above {here}; refusing to fall "
            f"back to a permissive message stub, which would accept fields the real message "
            f"rejects and pass the defect this check exists to find")
    types_ = {}
    for f in sorted(msg_dir.glob("*.msg")):
        fields = {}
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if not line or "=" in line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                kind, name = parts[0], parts[1]
                if kind.endswith("[]"):
                    fields[name] = list
                elif kind == "string":
                    fields[name] = str
                elif kind.startswith(("float", "double")):
                    fields[name] = float
                elif kind.startswith(("int", "uint")):
                    fields[name] = int
                elif kind == "bool":
                    fields[name] = bool
                else:
                    fields[name] = Any            # a nested message
        fields["header"] = Any

        def _init(self, _f=fields):
            # A generated message arrives with every field initialised; reading one before
            # writing it must not raise, or the stub fails where the real type would not.
            for n, kind in _f.items():
                object.__setattr__(self, n, kind())
        ns = {"__slots__": tuple(fields), "__init__": _init}
        types_[f.stem] = type(f.stem, (object,), ns)
    return types_


def install():
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    # rclpy.node.Node is subclassed, so it must be a real class, not a stub instance.
    import rclpy.node
    rclpy.node.Node = type("Node", (object,), {"__init__": lambda self, *a, **k: None,
                                               "__getattr__": lambda self, n: Any()})
    import lost3dsg.msg
    for name, cls in _real_msg_types().items():
        setattr(lost3dsg.msg, name, cls)
