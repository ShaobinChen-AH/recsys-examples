#!/usr/bin/env python3
"""
HotState Budget Enforcement Verification Script

This script checks whether the backend systems actually support the enforcement
mechanisms that HotState relies on. Run this BEFORE conducting experiments.

Usage:
    python verify_enforcement.py --checkpoint_dir <path> --gin_config <path>
"""

import argparse
import sys
import torch


def check_enforcement_support(model_path, config_path):
    """Check if enforcement mechanisms exist in the backend."""

    print("=" * 80)
    print("HotState Enforcement Verification")
    print("=" * 80)

    # Import necessary modules
    try:
        import gin
        from modules.hotstate.hotstate_controller import HotStateController
        from modules.hotstate.physical_budget import get_hbm_budget_bytes, get_hotstate_enabled
        from utils import DatasetArgs, NetworkArgs, RankingArgs  # Import gin-configurable classes
        print("[OK] Successfully imported HotState modules")
    except ImportError as e:
        print(f"[FAIL] Failed to import HotState modules: {e}")
        return False

    # Parse config
    try:
        gin.parse_config_file(config_path)
        print(f"[OK] Parsed config: {config_path}")
    except Exception as e:
        print(f"[FAIL] Failed to parse config: {e}")
        return False

    # Check if HotState is enabled
    hotstate_enabled = get_hotstate_enabled()
    total_hbm_bytes = get_hbm_budget_bytes()

    print(f"\n[Config Check]")
    print(f"  HotState Enabled: {hotstate_enabled}")
    print(f"  Total HBM Budget: {total_hbm_bytes / (1024**3):.3f} GiB")

    if not hotstate_enabled or total_hbm_bytes <= 0:
        print("\n[SKIP] HotState is not enabled or budget is not set")
        print("To enable HotState, set in your config:")
        print("  HotStateConfig.enabled = True")
        print("  HotStateConfig.total_hbm_budget_gib = <value>")
        return False

    # Load a minimal model to check backend support
    print(f"\n[Model Loading]")
    print(f"  Checkpoint: {model_path}")

    try:
        # This is a simplified check - adjust based on actual model loading
        # The key is to get access to the embedding and KV modules
        from inference.inference_gr_ranking import get_inference_hstu_model
        from inference.inference_gr_ranking import get_inference_dataset_and_embedding_configs

        dataset_args, emb_configs = get_inference_dataset_and_embedding_configs()

        # Create model with minimal config
        with torch.inference_mode():
            model, hotstate_controller = get_inference_hstu_model(
                emb_configs=emb_configs,
                max_batch_size=1,
                num_contextual_features=0,
                total_max_seqlen=128,
                checkpoint_dir=model_path,
                use_kvcache=True,
                return_controller=True,
            )

        print("[OK] Model loaded with HotState controller")

    except Exception as e:
        print(f"[FAIL] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check embedding enforcement support
    print(f"\n{'='*80}")
    print("EMBEDDING ENFORCEMENT CHECK")
    print("=" * 80)

    emb_adapter = hotstate_controller.emb_adapter
    emb_module = emb_adapter._module

    if emb_module is None:
        print("[FAIL] No embedding module connected to adapter")
        embedding_enforcement_ok = False
    else:
        print(f"[OK] Embedding module type: {type(emb_module).__name__}")

        # Check if enforcement method exists
        has_updater = hasattr(emb_module, "update_hotstate_admission_policy")
        print(f"\nMethod Check:")
        print(f"  update_hotstate_admission_policy exists: {has_updater}")

        if has_updater:
            print("[OK] Embedding enforcement method exists")

            # Try calling it with dummy values
            try:
                emb_module.update_hotstate_admission_policy(
                    item_indices=[],
                    max_admitted_keys=100,
                    enabled=True,
                )
                print("[OK] Embedding enforcement method is callable")
                embedding_enforcement_ok = True
            except Exception as e:
                print(f"[WARN] Embedding enforcement method failed: {e}")
                embedding_enforcement_ok = False
        else:
            print("[FAIL] Embedding enforcement method MISSING")
            print("\nThis means:")
            print("  - HotState cannot enforce embedding budgets")
            print("  - Embedding system will ignore budget allocations")
            print("  - update_admission_policy() calls will be silently skipped")
            print("\nTo fix: Implement update_hotstate_admission_policy() in InferenceEmbedding")
            embedding_enforcement_ok = False

    # Check KV enforcement support
    print(f"\n{'='*80}")
    print("KV CACHE ENFORCEMENT CHECK")
    print("=" * 80)

    kv_adapter = hotstate_controller.kv_adapter
    kv_gpu_mgr = kv_adapter._gpu_mgr

    if kv_gpu_mgr is None:
        print("[FAIL] No KV GPU manager connected to adapter")
        kv_enforcement_ok = False
    else:
        print(f"[OK] KV GPU manager type: {type(kv_gpu_mgr).__name__}")

        # Check if enforcement method exists
        has_setter = hasattr(kv_gpu_mgr, "set_active_page_limit")
        has_getter = hasattr(kv_gpu_mgr, "get_active_page_limit")

        print(f"\nMethod Check:")
        print(f"  set_active_page_limit exists: {has_setter}")
        print(f"  get_active_page_limit exists: {has_getter}")

        if has_setter and has_getter:
            print("[OK] KV enforcement methods exist")

            # Try calling them
            try:
                current_limit = kv_gpu_mgr.get_active_page_limit()
                print(f"[OK] Current page limit: {current_limit} pages")

                # Try setting a new limit
                test_limit = max(256, current_limit // 2)
                kv_gpu_mgr.set_active_page_limit(test_limit)
                new_limit = kv_gpu_mgr.get_active_page_limit()

                if new_limit == test_limit:
                    print(f"[OK] Page limit enforcement works (set {test_limit}, got {new_limit})")
                    kv_enforcement_ok = True
                else:
                    print(f"[WARN] Page limit not enforced (set {test_limit}, got {new_limit})")
                    kv_enforcement_ok = False

                # Restore original limit
                kv_gpu_mgr.set_active_page_limit(current_limit)

            except Exception as e:
                print(f"[WARN] KV enforcement methods failed: {e}")
                import traceback
                traceback.print_exc()
                kv_enforcement_ok = False
        else:
            print("[FAIL] KV enforcement methods MISSING")
            print("\nThis means:")
            print("  - HotState cannot enforce KV cache budgets")
            print("  - KV cache will ignore budget allocations")
            print("  - set_page_limit() calls will be silently skipped")
            print("\nTo fix: Implement set_active_page_limit() and get_active_page_limit()")
            print("         in GPU KV cache manager")
            kv_enforcement_ok = False

    # Final verdict
    print(f"\n{'='*80}")
    print("FINAL VERDICT")
    print("=" * 80)

    if embedding_enforcement_ok and kv_enforcement_ok:
        print("[PASS] Budget enforcement is FULLY SUPPORTED")
        print("\nYou can proceed with HotState experiments.")
        print("The control plane decisions will actually be enforced by backends.")
        return True
    elif embedding_enforcement_ok or kv_enforcement_ok:
        print("[PARTIAL] Budget enforcement is PARTIALLY SUPPORTED")
        if embedding_enforcement_ok:
            print("  ✓ Embedding enforcement works")
        else:
            print("  ✗ Embedding enforcement broken")
        if kv_enforcement_ok:
            print("  ✓ KV enforcement works")
        else:
            print("  ✗ KV enforcement broken")
        print("\nWARNING: Experiments will show mixed results.")
        print("Only the working subsystem will respect budgets.")
        return False
    else:
        print("[FAIL] Budget enforcement is NOT SUPPORTED")
        print("\nCRITICAL ISSUE:")
        print("  - Neither embedding nor KV enforcement methods exist")
        print("  - HotState control plane will run but have ZERO effect")
        print("  - All control plane overhead, zero benefit")
        print("  - This explains why performance is worse than baseline")
        print("\nDO NOT RUN EXPERIMENTS until enforcement is implemented.")
        print("You are measuring pure overhead with no actual resource management.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify HotState budget enforcement support"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--gin_config_file",
        type=str,
        required=True,
        help="Path to gin config file",
    )

    args = parser.parse_args()

    # Run verification
    success = check_enforcement_support(args.checkpoint_dir, args.gin_config_file)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
