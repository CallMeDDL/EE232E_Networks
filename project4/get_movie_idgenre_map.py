import os

def load_data(filename):
	with open(filename,'r', encoding='latin-1') as file:
		content = file.readlines()

	return content	

filename1 = "movie_genre.txt"
movie_genre = load_data(filename1)
filename2 = "par1_sampled_data_v1/movie_idname_map.txt"

movie_idname_map = load_data(filename2)


def data_process(content):
	res = {}
	for line in content:
		ele1 = line[:-1].split("\t\t")[0]
		ele2 = line[:-1].split("\t\t")[1]
		res[ele1] = ele2
	return res

movie_genre_list = data_process(movie_genre)
movie_idname_list = data_process(movie_idname_map)


def get_Movie_Genre_file(movie_genre, movie_idname, outputfile):
	with open(outputfile,'w', encoding='latin-1') as file:
		print("start writing...")
		for key in movie_idname.keys(): # key here is the movie id
			if movie_idname[key] in list(movie_genre.keys()): # info avaliable
				#print("name:", " ",movie_idname[key])
				file.write(str(key) + u"\t" + movie_genre[movie_idname[key]] + u"\n")
			else:
				file.write(str(key) + u"\t" + "NAN" + u"\n")
	print("write success!")

outputfile = "par1_sampled_data_v1/movie_idgenre_map.txt"
get_Movie_Genre_file(movie_genre_list, movie_idname_list,outputfile)



