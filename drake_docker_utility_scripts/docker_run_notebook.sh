#! /bin/bash
if [ "$#" != "2" ]; then 
	echo "Please supply two arguments: a Drake release (drake-YYYYMMDD), and a relative path to your preferred notebook directory"
	exit 1
else
	echo -e "\033[32m" Using $(pwd)$2 as your notebook root directory. "\033[0m"	
	docker pull mit6881/drake-course:$1
	docker run -it -p 8080:8080 -p 7000-7010:7000-7010 --rm -v "$(pwd)/$2"":"/psets \
		mit6881/drake-course:$1 /bin/bash -c "cd /psets && jupyter notebook --ip 0.0.0.0 --port 8080 --allow-root --no-browser"
fi
