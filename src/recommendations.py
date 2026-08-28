WASTE_GUIDE = {
    "cardboard": {
        "examples": ["shipping boxes", "cereal boxes", "paperboard packaging"],
        "recycling": "Flatten clean, dry cardboard before recycling.",
        "contamination": "Grease, food residue, and wet fibers can make cardboard non-recyclable.",
        "disposal": "Remove tape or plastic inserts where practical; follow local collection rules.",
    },
    "glass": {
        "examples": ["bottles", "jars", "clear or colored containers"],
        "recycling": "Rinse containers and separate glass only if your local program requires it.",
        "contamination": "Ceramics, mirrors, light bulbs, and broken mixed glass often need special handling.",
        "disposal": "Wrap broken glass safely and check local drop-off rules.",
    },
    "metal": {
        "examples": ["aluminum cans", "steel food cans", "clean foil"],
        "recycling": "Empty and rinse cans before recycling.",
        "contamination": "Paint cans, aerosols, and chemical containers may be hazardous.",
        "disposal": "Use household hazardous-waste programs for pressurized or chemical containers.",
    },
    "paper": {
        "examples": ["office paper", "newspaper", "magazines", "mail"],
        "recycling": "Recycle clean and dry paper in the paper stream.",
        "contamination": "Tissues, waxed paper, and food-soiled paper are usually not recyclable.",
        "disposal": "Shred sensitive documents; recycle only accepted paper types.",
    },
    "plastic": {
        "examples": ["bottles", "containers", "rigid packaging"],
        "recycling": "Check local rules and resin codes because plastic acceptance varies widely.",
        "contamination": "Film, foam, food residue, and mixed-material packaging often reduce recyclability.",
        "disposal": "Empty containers and keep caps only if your local program accepts them.",
    },
    "trash": {
        "examples": ["food-soiled packaging", "mixed materials", "non-recyclable waste"],
        "recycling": "Trash predictions usually mean the item is unlikely to fit a basic recycling stream.",
        "contamination": "Putting trash in recycling can contaminate otherwise recyclable material.",
        "disposal": "Use general waste unless local guidance provides a special disposal path.",
    },
}


def get_recommendation(class_name: str) -> str:
    guide = WASTE_GUIDE.get(class_name)
    if not guide:
        return "Follow local waste-management guidance for this item."
    return f"{guide['recycling']} Recycling rules vary by location."
