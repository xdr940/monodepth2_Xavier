class Base():
    def __init__(self,running_on,dev):
        super(Base, self).__init__()
        self.running_on = running_on
        self.dev= dev
    def get_fps(self):
        pass
        #print('base fps')


class Up(Base):
    def __init__(self, *args, **kwargs):
        super(Up, self).__init__(*args, **kwargs)
        print(self.dev,self.running_on)

    def get_fps(self):
        print('up fps')


up = Up(dev='cuda',running_on='pc')
