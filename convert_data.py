from numpy import std,mean

base = '/home/tzoor/Desktop/yoav/data/'

try:
    mmpi = open(base+'MEELMMPI.DAT')
    jud = open(base+'MEELJUD.DAT')
    con = open(base+'converted_data.txt','w')

    mmpi_lines = mmpi.readlines()
    jud_lines = jud.readlines()

    dat = []
    for k in range(len(mmpi_lines)):
        mmpi_outline = mmpi_lines[k][1:34]
        jud_outline = jud_lines[k][9:69]
        
        jud_data = [(int(jud_outline[2*i:2*i+2])-1)/10.0 for i in range(0,29)]
        jud_avg = sum(jud_data)/len(jud_data)
        crt = int(jud_outline[-1])-1
        
        data = [int(mmpi_outline[0:2])]
        data += [int(mmpi_outline[3*i-1:3*i+2]) for i in range(1,11)]
        data += [jud_avg,crt]
        dat.append(data)

    avg,sd = [],[]
    for i in range(len(dat[0][:-2])):
        arr = [dat[j][i] for j in range(len(dat))]
        avg.append(mean(arr))
        sd.append(std(arr))
        
    print avg
    print sd
        
    for i in range(len(dat[0][:-2])):
        M = max([dat[j][i] for j in range(len(dat))])
        m = min([dat[j][i] for j in range(len(dat))])
        for j in range(len(dat)):
            dat[j][i] = round(float(dat[j][i]-m)/(M-m),4)

    for data in dat:
        st = ''
        for el in data:
            st += str(el)+'\t'
        st += '\n'
        con.write(st)
except IOError:
    print "file error"
finally:
    mmpi.close()
    jud.close()
