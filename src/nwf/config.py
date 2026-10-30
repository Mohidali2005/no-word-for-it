MODELS = {
    "qwen":{"repo":"Qwen/Qwen3-VL-2B-Instruct","revision":"89644892e4d85e24eaac8bacfd4f463576704203"},
    "gemma":{"repo":"google/gemma-3-4b-it","revision":"093f9f388b31de276ce2de164bdc2081324b9767"},
    "llava":{"repo":"llava-hf/llava-1.5-7b-hf","revision":"b234b804b114d9e37bb655e11cbbb5f5e971b7a9"},
    # the released name taught and task trained checkpoints hold weights only so their processor comes from the base model
    "qwen_names":{"repo":"Lancelot53/qwen3-vl-2b-instruct-ordinary-names","revision":"d4b5086ade03fd8fc5805586d03f49e7edd346f1","processor":"qwen"},
    "qwen_task":{"repo":"Lancelot53/qwen3-vl-2b-instruct-task-finetuned","revision":"b2022d30302e94be141018006860f0db46503051","processor":"qwen"},
    "gemma_names":{"repo":"Lancelot53/gemma3-4b-it-ordinary-names","revision":"b24887f551cfb4eaeec6d4a9a96e17eeff3e5a9b","processor":"gemma"},
    "gemma_task":{"repo":"Lancelot53/gemma3-4b-it-task-finetuned","revision":"9c7e2891b13d9c7cecf8603ccd8dfa0617937cd8","processor":"gemma"},
}

AUTHORS_REPO = "https://github.com/Patchwork53/VLMs-Need-Words-COLM2026.git"
AUTHORS_COMMIT = "18655d407bbe69864eea3d2f13bba0144c8f4e33"

# the exact question and answer start used by the authors for the shape task
PROMPT = "Which shape in the second image is most similar to the REF shape in the first image?\nSelect from the following choices.\n(A) Point A\n(B) Point B\n(C) Point C\n(D) Point D\n"
PREFILL = "The correct option is ("
LETTERS = ["A","B","C","D"]

# every shape is drawn on a 200 pixel canvas like the authors squiggles with a 4 pixel line
CANVAS = 200
SUPERSAMPLE = 4
STROKE = 4
INK = (20,20,20)
# shapes are scaled so their longer side is 180 pixels which matches the squiggles the authors drew
SHAPE_BOX = 180

ITEMS = 600
# the first items of every set only pick the probe layer and never enter the evidence tests
CALIBRATION = 100
POOL = 40

# each set has its own seed so rebuilding one set never changes another
SETS = {
    "rep_named":{"kind":"rep","folder":"basic_shapes_TEST","shapes":"basic_shapes","named":1,"filled":1},
    "rep_squig30":{"kind":"rep","folder":"squiggles_30_TEST","shapes":"squiggles_30","named":0,"filled":0},
    "named_fill":{"kind":"named","seed":101,"named":1,"filled":1},
    "named_line":{"kind":"named","seed":102,"named":1,"filled":0},
    "unnamed_line":{"kind":"squiggle","anchors":30,"seed":103,"named":0,"filled":0},
    "unnamed_fill":{"kind":"blob","seed":104,"named":0,"filled":1},
    "named_cx":{"kind":"icon","seed":105,"named":1,"filled":1},
    "squig_n5":{"kind":"squiggle","anchors":5,"seed":106,"named":0,"filled":0},
    "squig_n10":{"kind":"squiggle","anchors":10,"seed":107,"named":0,"filled":0},
    "squig_n20":{"kind":"squiggle","anchors":20,"seed":108,"named":0,"filled":0},
    "squig_n50":{"kind":"squiggle","anchors":50,"seed":109,"named":0,"filled":0},
    "taught30":{"kind":"authors","folder":"squiggles_30","seed":110,"named":0,"filled":0},
    "untaught30":{"kind":"authors","folder":"squiggles_30_v2","seed":111,"named":0,"filled":0},
}
