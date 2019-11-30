import pickle
verbs_file = "morphs.txt"

def expand_dict(d):
	result = {}
	for key in d:
		if key in result:
			result[key] = result[key].union(d[key].difference({key}))
		else:
			result[key] = d[key].difference({key})
		for item in d[key]:
			if item in result:
				if item != key:
					result[item] = result[item].union(d[key].difference({item})).union({key})
				else:
					result[item] = result[item].union(d[key].difference({item}))
			else:
				if item != key:
					result[item] = d[key].difference({item}).union({key})
				else:
					d[key].difference({item})

	
	for key in result:
		result[key]=list(result[key])
	return result


with open(verbs_file,"r") as ip_file:
	ip_lines = ip_file.readlines()
	words = {}
	for line in ip_lines:
		line = line.strip().split()
		if len(line) != 3:
			print(line)
		word = line[1]
		word_form = line[0]
		if word in words:
			words[word].add(word_form)
		else:
			words[word]={word_form}


result = expand_dict(words)
pickle.dump(result,open("verbs.p","wb"))