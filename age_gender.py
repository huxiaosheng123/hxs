#computeing accuracy
def complete(olist,glist):
    print(len(olist))
    print(len(glist))
    num = 0
    if len(olist) == len(glist):
        for i in range(len(olist)):
            oname = olist[i][0]
            gname = glist[i][0]
            ovalue = olist[i][1]
            gvalue = glist[i][1]
            # print(oname)
            # print(gname)
            print('######################')
            if oname == gname:
                if ovalue == gvalue:
                    num += 1
    print(num)
    print(num/len(olist))

def complete_age(olist,glist):
    mid_num = 0
    gmidnum = 0
    old_num = 0
    goldnum = 0
    child_num = 0
    gchildnum = 0
    if len(olist) == len(glist):
        for i in range(len(olist)):
            oname = olist[i][0]
            gname = glist[i][0]
            ovalue = olist[i][1]
            gvalue = glist[i][1]
            print(oname)
            print(gname)
            print(ovalue)
            print(gvalue)
            print('##############')
            if oname == gname:
                if ovalue == 'age_middle':
                    mid_num += 1
                    if gvalue == 'age_middle':
                        gmidnum += 1
                        print('11111111111111111111111111111111111')
                if ovalue == 'age_old':
                    old_num += 1
                    if gvalue == 'age_old':
                        goldnum += 1
                if ovalue == 'age_child':
                    child_num += 1
                    if gvalue == 'age_child':
                        gchildnum += 1
    print(gchildnum)
    print(mid_num)
    print('middle:',gmidnum/mid_num)
    print('old:',goldnum/old_num)
    print('child:',gchildnum/child_num)


original = r'E:\PycharmProjects\Real-Time-Voice-Cloning\metrics\yuanshiresult.txt'
generate = r'F:\shengchengresult.txt'

agelist = []
genderlist = []
with open(original) as ori_f:
    oline = ori_f.readlines()
    for oline1 in oline:
        oline2 = oline1.strip().split(' ')
        #print(oline2)
        try:
            name = oline2[2]
        except:
            name = 'None'
        try:
            age = oline2[0]
        except:
            age = 'None'
        try:
            gender = oline2[1]
        except:
            gender = 'None'
        agelist.append([name,age])
        genderlist.append([name,gender])
    # print(agelist)
    # print(genderlist)


agelist_g = []
genderlist_g = []
with open(generate) as gen_f:
    gline = gen_f.readlines()
    for gline1 in gline:
        gline2 = gline1.strip().split(' ')
        #print(oline2)
        try:
            name = gline2[2]
        except:
            name = 'None'
        try:
            age = gline2[0]
        except:
            age = 'None'
        try:
            gender = gline2[1]
        except:
            gender = 'None'
        agelist_g.append([name,age])
        genderlist_g.append([name,gender])
    # print(agelist_g)
    # print(genderlist_g)

complete_age(agelist,agelist_g)

