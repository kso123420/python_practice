import requests
from bs4 import BeautifulSoup

class job:

    def __init__(self, title, company, position, region, job_url):
        self.title = title
        self.company = company
        self.position = position
        self.region = region
        self.job_url = job_url

class webpage:

    def __init__(self, url):
        self.url = url
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        self.pages = len(soup.find("div", class_="pagination").find_all("span", class_="page"))
        self.jobs = []
    
    def find_jobs(self):
        for x in range(self.pages):
            modified_url = self.url.replace("page=1", f"page={x+1}")
            response = requests.get(modified_url)
            soup = BeautifulSoup(response.content, "html.parser", )
            jobs = soup.find("section", class_="jobs", ).find_all("li")[0:-1]
            
            for job_data in jobs:
                title = job_data.find("span", class_="title").text
                company = job_data.find_all("span", class_="company")[0].text
                position = job_data.find_all("span", class_="company")[1].text
                region = job_data.find_all("span", class_="company")[2].text
                job_url = f"https://weworkremotely.com{job_data.find_all("a")[1]["href"]}"
                self.jobs.append(job(title, company, position, region, job_url))

        print(f"{len(self.jobs)} jobs was found.")

    def print_jobs(self):
        for job in self.jobs:
            print(f"(Title: {job.title}, Company: {job.company}, Position: {job.position}, Region: {job.region}, URL: {job.job_url})")

wwr = webpage("https://weworkremotely.com/remote-full-time-jobs?page=1")
wwr.find_jobs()
wwr.print_jobs()