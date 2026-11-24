"""launch_v5.py - viz2 v5 (2026-11-23): runs env_check/scripts/step2b_mpc_isaac.py UNCHANGED (as gh_field/launch/step2b_mpc_isaac_field.py),
but the module g1_viz is loaded from this folder's g1_viz_v5.py (import hook; the original goal1/scripts/g1_viz.py is not changed).
The file name does not contain "step2b_mpc_isaac"; the run script holds the shared GPU lock. Own option (removed before step2b parses argv):
  --v5-env NAME=VALUE (repeatable): set in os.environ here (run_isaac_ros.sh starts Isaac with env -i). All other arguments go to step2b."""
import importlib.util
import os
import runpy
import sys

S2B = os.path.expanduser(os.environ.get("EFFMASS_ROOT", "~/effmass_mpc") + "/env_check/scripts/step2b_mpc_isaac.py")
V5 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "g1_viz_v5.py")


class _Hook:   # g1_viz -> g1_viz_v5.py; g1_human_link: person capsules grey (display constants only, after the module ran)
    def find_spec(self, name, path=None, target=None):
        if name == "g1_viz":
            return importlib.util.spec_from_file_location("g1_viz", V5)
        if name != "g1_human_link" or getattr(self, "_hl", False):
            return None
        self._hl = True; spec = importlib.util.find_spec(name)
        if spec is None:
            return None
        ex = spec.loader.exec_module

        def exec_module(m):
            ex(m); m.CAPSULE_RGB = tuple(float(x) for x in os.environ.get("V5_PERSON_RGB", "0.62,0.62,0.62").split(",")); m.CAPSULE_OPACITY = float(os.environ.get("V5_PERSON_OPACITY", "0.85"))
        spec.loader.exec_module = exec_module
        return spec


rest = []; a = sys.argv[1:]; i = 0
while i < len(a):
    if a[i] == "--v5-env":
        k, v = a[i + 1].split("=", 1); os.environ[k] = v; i += 2; continue
    rest.append(a[i]); i += 1
sys.meta_path.insert(0, _Hook())
sys.argv = [S2B] + rest
runpy.run_path(S2B, run_name="__main__")
