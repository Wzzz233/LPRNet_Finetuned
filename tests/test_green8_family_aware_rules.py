#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_EVAL = ROOT / 'src' / 'evaluation'
if str(SRC_EVAL) not in sys.path:
    sys.path.insert(0, str(SRC_EVAL))

import eval_lpr_detailed as detailed
import eval_pos0_fusion as pos0


def test_green8_d_or_f_third_position_remains_valid():
    assert detailed.green_full_valid('沪ADF1234')
    assert detailed.green_full_valid('苏AF9X8K7')
    assert pos0.green_full_valid('沪ADF1234')
    assert pos0.green_full_valid('苏AF9X8K7')


def test_green8_non_df_third_position_is_valid_without_forcing_last_df():
    # 新能源规则校正：高保有量城市第三位不一定是 D/F；
    # 若第三位是 A/B/C/E/G/H/J/K 等，末位也不应被 beam 强制为 D/F。
    examples = ['陕AA02222', '皖TAF4D7H', '皖HHF8N5Q', '沪AB12345', '苏AG9K8Q7']
    for text in examples:
        assert detailed.green_prefix_valid(text), text
        assert detailed.green_full_valid(text), text
        assert pos0.green_prefix_valid(text), text
        assert pos0.green_full_valid(text), text


def test_green8_rejects_invalid_non_green_characters_and_lengths():
    invalid = ['陕AA0222', '陕AA022222', '陕A-02222', '陕AI02222']
    for text in invalid:
        assert not detailed.green_full_valid(text), text
        assert not pos0.green_full_valid(text), text
