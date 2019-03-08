from .dssm import DSSM
from .cdssm import CDSSM
from .arci import ArcI
from .arcii import ArcII
from .match_pyramid import MatchPyramid
from .duet import DUET
from .mvlstm import MVLSTM
from .conv_highway import ConvHighway

import matchzoo
def list_available():
    return matchzoo.engine.BaseModel.__subclasses__()
