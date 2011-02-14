git checkout master
git pull
cd doc
make clean
make html
rsync -av build/html/ root@beli:/var/www/natter/doc
cd ..
git archive -o natter.zip HEAD
rsync -av natter.zip root@beli:/var/www/natter/