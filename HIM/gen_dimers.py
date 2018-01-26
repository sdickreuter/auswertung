import numpy as np
import pandas as pd



gridx = np.linspace(-9,9,4)
gridy = np.linspace(-9,9,4)

gridx, gridy = np.meshgrid(gridx,gridy)
gridx = gridx.ravel()
gridy = gridy.ravel()


gaps = np.array([5,10,15,20,30])*1e-3
width = 100*1e-3
height = 25*1e-3

for gap in gaps:

    x = gridx
    y = gridy

    x = np.hstack((x,x+gap+width))
    y = np.hstack((y,y))


    heights = np.repeat(height,len(x))
    widths = np.repeat(width, len(x))

    df_x = pd.Series(x)
    df_y = pd.Series(y)

    df = pd.DataFrame({'x' : df_x,'y' : df_y})

    #print(df)

    writer = pd.ExcelWriter('./dimer_'+str(int(gap*1e3))+'nm_gap.xls')
    df.to_excel(writer,index=False)
    #df_y.to_excel(writer,'y',index=False)
    writer.save()
