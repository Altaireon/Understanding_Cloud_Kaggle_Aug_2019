from .efficientnetb5_c import *

class EfficientNetB5Segmentation(nn.Module):

    def __init__(self, num_class=4, drop_connect_rate=0.2):
        super(EfficientNetB5Segmentation, self).__init__()

        e = EfficientNetB5(drop_connect_rate)
        self.stem   = e.stem
        self.block1 = e.block1
        self.block2 = e.block2
        self.block3 = e.block3
        self.block4 = e.block4
        self.block5 = e.block5
        self.block6 = e.block6
        self.block7 = e.block7
        self.last   = e.last
        e = None  #dropped

        #---
        self.lateral0 = nn.Conv2d(2048, 64,  kernel_size=1, padding=0, stride=1)
        self.lateral1 = nn.Conv2d( 176, 64,  kernel_size=1, padding=0, stride=1)
        self.lateral2 = nn.Conv2d(  64, 64,  kernel_size=1, padding=0, stride=1)
        self.lateral3 = nn.Conv2d(  40, 64,  kernel_size=1, padding=0, stride=1)

        self.top1 = nn.Sequential(
            ConvGnUp2d( 64, 64),
            ConvGnUp2d( 64, 64),
            ConvGnUp2d( 64, 64),
        )
        self.top2 = nn.Sequential(
            ConvGnUp2d( 64, 64),
            ConvGnUp2d( 64, 64),
        )
        self.top3 = nn.Sequential(
            ConvGnUp2d( 64, 64),
        )
        self.top4 = nn.Sequential(
            nn.Conv2d(64*3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.logit_mask = nn.Conv2d(64,1+num_class,kernel_size=1)

    def forward(self, x):
        batch_size,C,H,W = x.shape

        x = self.stem(x)            #; print('stem  ',x.shape)
        x = self.block1(x)    ;x0=x #; print('block1',x.shape)
        x = self.block2(x)    ;x1=x #; print('block2',x.shape)
        x = self.block3(x)    ;x2=x #; print('block3',x.shape)
        x = self.block4(x)          #; print('block4',x.shape)
        x = self.block5(x)    ;x3=x #; print('block5',x.shape)
        x = self.block6(x)          #; print('block6',x.shape)
        x = self.block7(x)          #; print('block7',x.shape)
        x = self.last(x)      ;x4=x #; print('last  ',x.shape)

        # segment
        t0 = self.lateral0(x4)
        t1 = upsize_add(t0, self.lateral1(x3)) #16x16
        t2 = upsize_add(t1, self.lateral2(x2)) #32x32
        t3 = upsize_add(t2, self.lateral3(x1)) #64x64

        t1 = self.top1(t1) #128x128
        t2 = self.top2(t2) #128x128
        t3 = self.top3(t3) #128x128

        t = torch.cat([t1,t2,t3],1)
        t = self.top4(t)
        logit_mask = self.logit_mask(t)
        logit_mask = F.interpolate(logit_mask, scale_factor=2.0, mode='bilinear', align_corners=False)

        return logit_mask
