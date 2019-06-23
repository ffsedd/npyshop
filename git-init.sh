git init

git add numpyshop.py
git add npimage.py
git add .gitignore
git commit -m "initial commit"
git status

# now create repository online at github.com


git remote add origin https://github.com/ffsedd/numpyshop

git pull origin branchname --allow-unrelated-histories
git push -u origin master
