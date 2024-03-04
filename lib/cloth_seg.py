from enum import IntEnum


class ClothSeg(IntEnum):
    BACKGROUND = 0
    HAT = 1
    HAIR = 2
    SUNGLASSES = 3
    UPPER_CLOTHES = 4
    SKIRT = 5
    PANTS = 6
    DRESS = 7
    BELT = 8
    LEFT_SHOE = 9
    RIGHT_SHOE = 10
    FACE = 11
    LEFT_LEG = 12
    RIGHT_LEG = 13
    LEFT_ARM = 14
    RIGHT_ARM = 15
    BAG = 16
    SCARF = 17


def everyhing_but_background_face_and_hair() -> list[ClothSeg]:
    return [
        t
        for t in ClothSeg
        if t not in [ClothSeg.BACKGROUND, ClothSeg.HAIR, ClothSeg.FACE]
    ]
