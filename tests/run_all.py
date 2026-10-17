"""Run all tests without pytest:  python -m tests.run_all   (pytest also works: python -m pytest)"""
import importlib, sys, time, traceback

failed = 0
for mod in ("tests.test_core", "tests.test_pipeline_and_api", "tests.test_features_v2"):
    m = importlib.import_module(mod)
    for name in sorted(n for n in dir(m) if n.startswith("test_")):
        t = time.time()
        try:
            getattr(m, name)()
            print(f"PASS {mod}.{name} ({time.time() - t:.1f}s)")
        except Exception:
            failed += 1
            print(f"FAIL {mod}.{name}")
            traceback.print_exc()
print("FAILED" if failed else "ALL PASSED", failed)
sys.exit(1 if failed else 0)
