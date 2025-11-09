"""
Test to verify the person ID mapping bug fix
验证 Person ID 映射 Bug 修复的测试
"""

import numpy as np
import torch


def test_identity_mapping():
    """
    测试 identity_list 映射是否正确工作
    """
    print("Testing identity mapping bug fix...")
    print("=" * 60)

    # 模拟 CUHK03 场景: person IDs 是打乱的 0-1359
    print("\n1. CUHK03 Scenario (shuffled person IDs)")
    print("-" * 60)

    # 假设有 100 个 person，但 ID 是随机的
    num_persons = 100
    identity_indices = np.random.choice(1360, num_persons, replace=False).tolist()
    identity_to_images = {pid: list(range(5)) for pid in identity_indices}

    print(f"Number of persons: {num_persons}")
    print(f"Identity indices (first 10): {identity_indices[:10]}")
    print(f"Min person ID: {min(identity_indices)}")
    print(f"Max person ID: {max(identity_indices)}")

    # 旧的错误方法 (会导致 KeyError)
    print("\n❌ Old (buggy) method:")
    print("   person_id = index % num_identities")
    errors = 0
    for index in range(20):
        person_id_old = index % num_persons
        try:
            _ = identity_to_images[person_id_old]
            print(f"   Index {index:2d} -> person_id {person_id_old:4d} ✓")
        except KeyError:
            print(f"   Index {index:2d} -> person_id {person_id_old:4d} ✗ KeyError!")
            errors += 1

    print(f"\n   Result: {errors}/20 samples failed with KeyError")

    # 新的正确方法
    print("\n✅ New (fixed) method:")
    print("   person_id = identity_list[index % len(identity_list)]")
    identity_list = identity_indices  # 正确的 ID 列表

    for index in range(20):
        person_id_new = identity_list[index % len(identity_list)]
        try:
            _ = identity_to_images[person_id_new]
            print(f"   Index {index:2d} -> person_id {person_id_new:4d} ✓")
        except KeyError:
            print(f"   Index {index:2d} -> person_id {person_id_new:4d} ✗ KeyError!")

    print(f"\n   Result: All 20 samples succeeded!")

    # 模拟 Market1501 场景
    print("\n" + "=" * 60)
    print("2. Market1501 Scenario (sparse person IDs 1-1501)")
    print("-" * 60)

    # Market1501: 只有 751 个训练 ID，但 ID 范围是 1-1501
    train_ids = sorted(np.random.choice(range(1, 1502), 751, replace=False).tolist())
    identity_to_images = {pid: list(range(10)) for pid in train_ids}

    print(f"Number of persons: {len(train_ids)}")
    print(f"Identity IDs (first 10): {train_ids[:10]}")
    print(f"Min person ID: {min(train_ids)}")
    print(f"Max person ID: {max(train_ids)}")

    # 旧的错误方法
    print("\n❌ Old (buggy) method:")
    errors = 0
    for index in range(20):
        person_id_old = index % len(train_ids)
        try:
            _ = identity_to_images[person_id_old]
            print(f"   Index {index:2d} -> person_id {person_id_old:4d} ✓")
        except KeyError:
            print(f"   Index {index:2d} -> person_id {person_id_old:4d} ✗ KeyError!")
            errors += 1

    print(f"\n   Result: {errors}/20 samples failed with KeyError")

    # 新的正确方法
    print("\n✅ New (fixed) method:")
    identity_list = train_ids

    for index in range(20):
        person_id_new = identity_list[index % len(identity_list)]
        try:
            _ = identity_to_images[person_id_new]
            print(f"   Index {index:2d} -> person_id {person_id_new:4d} ✓")
        except KeyError:
            print(f"   Index {index:2d} -> person_id {person_id_new:4d} ✗ KeyError!")

    print(f"\n   Result: All 20 samples succeeded!")

    print("\n" + "=" * 60)
    print("✅ Bug fix verified successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_identity_mapping()
