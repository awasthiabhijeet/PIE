# maps edits to their ids

import pickle
import utils

class Opcodes():
    def __init__(self,path_common_inserts,
                 path_common_multitoken_inserts,
                 use_transforms=True):
        USE_TRANSFORMS=use_transforms #turn this to false if use transforms are not be used
        print("path_common_inserts: {}".format(path_common_inserts))
        print("path_common_multitoken_inserts: {}".format(path_common_multitoken_inserts))

        self.UNK = 0 #dummy
        self.SOS = 1 #dummy
        self.EOS = 2 #dummy
        self.CPY = 3 #copy 
        self.DEL = 4 #delete

        APPEND_BEGIN = 5
        self.APPEND_BEGIN = APPEND_BEGIN    

        self.APPEND = {} #appends to their edit ids
        self.REP = {} #replaces to their edit ids
        APPEND = self.APPEND
        REP = self.REP

        common_inserts = pickle.load(open(path_common_inserts,"rb"))
        common_multitoken_inserts = pickle.load(open(path_common_multitoken_inserts,"rb"))

        for item in common_inserts:
            self.reg_append(item)
        for item in common_multitoken_inserts:
            self.reg_append(item)

        for item in common_inserts:
            self.reg_rep(item)    
        for item in common_multitoken_inserts:
            self.reg_rep(item)

        APPEND_END = APPEND_BEGIN + len(APPEND) - 1
        vocab_size = APPEND_BEGIN + len(APPEND)
        REP_BEGIN = APPEND_END + 1
        vocab_size += len(REP)
        REP_END = REP_BEGIN + len(REP) -1

        # TRANSFORM_SUFFIXES
        if USE_TRANSFORMS:
            self.APPEND_SUFFIX={}
            APPEND_SUFFIX = self.APPEND_SUFFIX 
            # APPEND_SUFFIX is different from APPEND
            # APPEND_SUFFIX maps suffix of APPEND-based-suffix-transformation to its edit id
            # This is used to map all the APPENDs to corresponding APPEND based suffix transformation
            self.APPEND_s = vocab_size
            APPEND_SUFFIX["##s"]=self.APPEND_s
            vocab_size += 1
            self.REMOVE_s = vocab_size
            vocab_size += 1

            self.APPEND_d = vocab_size
            APPEND_SUFFIX["##d"]=self.APPEND_d
            vocab_size += 1
            self.REMOVE_d = vocab_size
            vocab_size += 1

            self.APPEND_es = vocab_size
            APPEND_SUFFIX["##es"]=self.APPEND_es
            vocab_size += 1
            self.REMOVE_es = vocab_size
            vocab_size += 1

            self.APPEND_ing = vocab_size
            APPEND_SUFFIX["##ing"]=self.APPEND_ing
            vocab_size += 1
            self.REMOVE_ing = vocab_size
            vocab_size += 1

            self.APPEND_ed = vocab_size
            APPEND_SUFFIX["##ed"]=self.APPEND_ed
            vocab_size += 1
            self.REMOVE_ed = vocab_size
            vocab_size += 1

            self.APPEND_ly = vocab_size
            APPEND_SUFFIX["##ly"]=self.APPEND_ly
            vocab_size += 1
            self.REMOVE_ly = vocab_size
            vocab_size += 1

            self.APPEND_er = vocab_size
            APPEND_SUFFIX["##er"]=self.APPEND_er
            vocab_size += 1
            self.REMOVE_er = vocab_size
            vocab_size += 1

            self.APPEND_al = vocab_size
            APPEND_SUFFIX["##al"]=self.APPEND_al
            vocab_size += 1
            self.REMOVE_al = vocab_size
            vocab_size += 1

            self.APPEND_n = vocab_size
            APPEND_SUFFIX["##n"]=self.APPEND_n
            vocab_size += 1
            self.REMOVE_n = vocab_size
            vocab_size += 1

            self.APPEND_y = vocab_size
            APPEND_SUFFIX["##y"]=self.APPEND_y
            vocab_size += 1
            self.REMOVE_y = vocab_size
            vocab_size += 1

            self.APPEND_ation = vocab_size
            APPEND_SUFFIX["##ation"]=self.APPEND_ation
            vocab_size += 1
            self.REMOVE_ation = vocab_size
            vocab_size += 1

            self.E_TO_ING = vocab_size
            vocab_size += 1
            self.ING_TO_E = vocab_size
            vocab_size += 1

            self.D_TO_T = vocab_size
            vocab_size += 1
            self.T_TO_D = vocab_size
            vocab_size += 1

            self.D_TO_S = vocab_size
            vocab_size += 1
            self.S_TO_D = vocab_size
            vocab_size += 1

            self.S_TO_ING = vocab_size
            vocab_size += 1
            self.ING_TO_S = vocab_size
            vocab_size += 1

            self.N_TO_ING = vocab_size
            vocab_size += 1
            self.ING_TO_N = vocab_size
            vocab_size += 1

            self.T_TO_NCE = vocab_size
            vocab_size += 1
            self.NCE_TO_T = vocab_size
            vocab_size += 1

            self.S_TO_ED = vocab_size
            vocab_size += 1
            self.ED_TO_S = vocab_size
            vocab_size += 1

            self.ING_TO_ED = vocab_size
            vocab_size += 1
            self.ED_TO_ING = vocab_size
            vocab_size += 1

            self.ING_TO_ION = vocab_size
            vocab_size += 1
            self.ION_TO_ING = vocab_size
            vocab_size += 1

            self.ING_TO_ATION = vocab_size
            vocab_size += 1
            self.ATION_TO_ING = vocab_size
            vocab_size += 1

            self.T_TO_CE = vocab_size
            vocab_size += 1
            self.CE_TO_T = vocab_size
            vocab_size += 1

            self.Y_TO_IC = vocab_size
            vocab_size += 1
            self.IC_TO_Y = vocab_size
            vocab_size += 1

            self.T_TO_S = vocab_size
            vocab_size += 1
            self.S_TO_T = vocab_size
            vocab_size += 1

            self.E_TO_AL = vocab_size
            vocab_size += 1
            self.AL_TO_E = vocab_size
            vocab_size += 1

            self.Y_TO_ILY = vocab_size
            vocab_size += 1
            self.ILY_TO_Y = vocab_size
            vocab_size += 1

            self.Y_TO_IED = vocab_size
            vocab_size += 1
            self.IED_TO_Y = vocab_size
            vocab_size += 1

            self.Y_TO_ICAL = vocab_size
            vocab_size += 1
            self.ICAL_TO_Y = vocab_size
            vocab_size += 1

            self.Y_TO_IES = vocab_size
            vocab_size += 1
            self.IES_TO_Y = vocab_size
            vocab_size += 1
    

    def reg_append(self,word):
        #registers an APPEND
        if word not in self.APPEND:
            self.APPEND[word] = len(self.APPEND) + self.APPEND_BEGIN
        else:
            print("Skipping duplicate opcode", word)

    def reg_rep(self,word):
        #registers a REPLACE
        if word not in self.REP:
            self.REP[word] = len(self.REP) + len(self.APPEND) + self.APPEND_BEGIN
        else:
            print("Skipping duplicate opcode", word)
