from facelib.data import LFWDataset, FaceImageFolderDataset


lfw = LFWDataset(split='all', aligned=False)

test = FaceImageFolderDataset(name='test')

print(len(test))

print(len(lfw))

print(lfw[0])

print(lfw.get_n_identities())
print(lfw.get_n_images())


from facelib.modules import LinearHeader

header = LinearHeader(1, 1)
