
class Plate :
    listDataCharacters=None
    probability=0
    formatPlate=""
    patterns = ["ccnnnnc","ccnnnncc","ccccccc","ccnnnnn","ccnnnnnn","cccnnnnc","ccnnnnc","cnnnnccc"]
    matchedPattern=None
    
    def __init__(self,_listDataCharacters):
        self.listDataCharacters=_listDataCharacters
        for dc in self.listDataCharacters:
            self.probability+=dc.confidence
            if(ord(dc.characterOCR) == 48):
                self.formatPlate+="x"
            else:
                self.formatPlate += "c" if ord(dc.characterOCR) >64 else "n" 
        
        self.probability= self.probability/len(self.listDataCharacters)
        
        
    
    def plausible(self):
        
        return any(self.comparator(self.formatPlate,p) for p in self.patterns)
    
    
    
    def comparator (self, formatPlate, pattern):
        flag = True
        
        if (len(formatPlate) == len(pattern)):
            
            for index in range(len(pattern)):
                if(not(pattern[index]==formatPlate[index] or formatPlate[index]=='x')):
                    flag=False
                    break;
                    
        else:
            flag= False        
            
        if(flag):
            self.matchedPattern=pattern
        
        return flag
    
    
    def __str__(self):
        label=""
        
        for c in self.listDataCharacters:
            label+=c.characterOCR
        
        return label
    
    def getProability(self):
        return self.probability