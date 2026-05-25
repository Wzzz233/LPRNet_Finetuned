import random
from dataclasses import dataclass
from pathlib import Path

from generate_multi_plate import MultiPlateGenerator
from plate_number import digits, letters, provinces


@dataclass(frozen=True)
class RenderResult:
    image_bgr: object
    text: str
    family: str
    board_style: str
    is_double: bool


class SpecialPlateRenderer:
    def __init__(self, root_dir):
        root = Path(root_dir)
        self._generator = MultiPlateGenerator(str(root / "plate_model"), str(root / "font_model"))
        self._rng = random.Random(20260517)

    def render_plate(self, text, family):
        board_style, is_double = self._family_to_board(family)
        image = self._generator.generate_plate_special(text, board_style, is_double)
        return RenderResult(
            image_bgr=image,
            text=text,
            family=family,
            board_style=board_style,
            is_double=is_double,
        )

    def sample_text(self, family, province=None):
        if family == "yellow_single":
            if province is None:
                province = self._rng.choice(provinces)
            return province + self._rng.choice(letters) + "".join(
                self._rng.choice(digits + letters) for _ in range(5)
            )
        if family == "police":
            if province is None:
                province = self._rng.choice(provinces)
            return province + self._rng.choice(letters) + "".join(
                self._rng.choice(digits + letters) for _ in range(4)
            ) + "警"
        if family == "embassy":
            return "使" + "".join(self._rng.choice(digits) for _ in range(6))
        raise ValueError(f"Unsupported family: {family}")

    @staticmethod
    def _family_to_board(family):
        if family == "yellow_single":
            return "yellow", False
        if family == "police":
            return "white", False
        if family == "embassy":
            return "black_shi", False
        raise ValueError(f"Unsupported family: {family}")
