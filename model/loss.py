from model.losses.losses import LC, LI, LRec


class OurLosses():
    def __init__(self):
        super(OurLosses, self).__init__()
        self.l_c = LC()
        self.l_i = LI()
        self.l_rec = LRec()
