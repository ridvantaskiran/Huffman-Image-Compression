# Huffman-Image-Compression
A Image and a txt file compression program that compresses the .png or .txt files with user friendly GUI. The user can select the method which are gray level or colored images and txt files. 

It has also difference method. In this method, input array was used by taking the difference instead of using it directly. The first column is kept the same, and the other columns are subtracted from each other with the previous one. The same procedure is used for the first column, the first index kept for a pivot index, and the other indexes that are in the first column subtracted from each other with the previous one. Since the goal is to reduce the space, with only holding pivot and differenced NumPy image array, we can minimize the storage


![image](https://user-images.githubusercontent.com/59413074/187638518-4de3992e-8c97-4505-8189-35422c7cab30.png)

![image](https://user-images.githubusercontent.com/59413074/187638532-1f3e7f9d-6b4a-4d2c-8b15-573d2f3c4b5a.png)

