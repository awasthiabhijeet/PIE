import re

reg_ex = re.compile(r"^[a-z][a-z]*[a-z]$")
no_reg_ex = re.compile(r".*[0-9].*")
mc_reg_ex = re.compile(r".*[A-Z].*[A-Z].*")

def containsNumber(text):
  return no_reg_ex.match(text)

def containsMultiCapital(text):
  return mc_reg_ex.match(text)

def can_spellcheck(w: str):
    #return not ((not reg_ex.match(w)) or containsMultiCapital(w) or containsNumber
    if reg_ex.match(w):
    	return True
