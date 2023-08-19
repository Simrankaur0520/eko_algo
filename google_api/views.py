from import_statements import *
from django.db.models import Q
#----------------------------Annotation Functions--------------------------------------
class roundRating(Func):
    function = 'ROUND'
    template='%(function)s(%(expressions)s, 1)'
class twoDecimal(Func):
    function = 'ROUND'
    template='%(function)s(%(expressions)s, 2)'
class Round(Func):
    function = 'ROUND'
    template='%(function)s(%(expressions)s, 0)'


@api_view(['GET'])
def test_call(request):
    return Response("hello")


@api_view(['POST'])
def login(request):
    data=request.data
    u_email=data["email"]
    password=data["password"]

    if "@" in u_email:
        email=u_email
        try:
            user=user_login.objects.filter(email=email,password=password).values("id","name","username","email")[0]
            res={
            "status":True,
            "message":"Login Successful",
            "status_code":200,
            "data":user
        }
        except:
            res={
                "status":False,
                "message":"Something went wrong !"
            }
    else:
        username=u_email
        try:
            user=user_login.objects.filter(username=username,password=password).values("id","name","username","email")[0]
            res={
            "status":True,
            "message":"Login Successful",
            "status_code":200,
            "data":user
        }
        except:
            res={
                "status":False,
                "message":"Something went wrong !"
            }
    

    return Response(res)
# ------------------------------- Dashboard---------------------------------------
@api_view(['POST'])
def dashboard(request):
    data=request.data
    client_id=data['client_id']
    res={}
    all_branch_data={}
    specific_branch_data={}
    client_analysis={}
    client_sentimental_analysis={}
    client_analysis_list=[]
    competitor_analysis={}
    competitor_sentimental_analysis={}
    competitor_analysis_list=[]

    #-------------------------------fetching data from databse----------------------------------
    rating_data=ratings.objects.filter(client_id=client_id).values('source_name','rating')
    review=reviews.objects.filter(client_id=client_id).values()
    comp_reviews=competitor_reviews.objects.filter(competitor_branch_id="ABSN01").values()
    #------------------------------------------stats data--------------------------------------
    
    all_branch_data['branch_name']="Overall"
    specific_branch_data['stats']=rating_data
    all_branch_data['specific_branch_data']=specific_branch_data

    swot_data= [
          {
            "title": "Strength",
            "bullet_points": [
              "Tasty foods",
              "Cost effective"
            ]
          },
          {
            "title": "Recomendation",
            "bullet_points": [
              "Employee training",
              "Improve food taste",
              " Improve the cleanliness and hygiene"
            ]
          },
          {
            "title": "Weakness",
            "bullet_points": [
              "Hygine ",
              "Location"
            ]},
        {
            "title": "Competitor Analysis",
            "bullet_points": [
              "Srinidhi Sagar has a higher percentage of positive comments. ",
              "But Kadamba veg shows a lower percentage of negative comments."
            ]}
    ]
    actions_required=["Promotional Discounts","Improve cleanliness and maintenance","Introduce staff training Program","Hire better chefs"]
    topics=["Hygiene","Room Maintenance","Staff service and Training","Food quality"]
    #---------------------------rating over sentiment and time --------------------------------
    #client analysis res
    gr_obj = review.order_by('-date').values('date__year', 'date__month')\
                                                     .annotate(
                                                                survey_date = F('date'),
                                                                value = twoDecimal(Avg('rating')),
                                                                                                                                  )
    gr_obj = pd.DataFrame(gr_obj)
    gr_obj['title'] = gr_obj['survey_date'].apply(lambda x : datetime.strptime(str(x)[:10],'%Y-%m-%d').strftime('%b-%Y'))
    gr_obj = gr_obj.to_dict(orient='records')
    client_analysis['heading']="Rating Trend"
    client_analysis["graph_type"]="bar"
    client_analysis['source']="Google"
    client_analysis['graph_data']=gr_obj
    
    client_analysis_list.append(client_analysis)
    
    
    
    count= review.count()
    positive=review.filter(sentiment="Positive").count()
    negative=review.filter(Q(sentiment="Extreme") | Q(sentiment="Negative")).count()
    positive_percentage=round(((positive/count)*100),2)
    negative_percentage=round(((negative/count)*100),2)
    client_sentimental_analysis['heading']="sentimental analysis"
    client_sentimental_analysis["graph_type"]="bar"
    client_sentimental_analysis['source']="Google"
    graph_data=[{
                "title": "Positive",
                "value": positive_percentage
              },
              {
                "title": "Negative",
                "value": negative_percentage
              }]
    client_sentimental_analysis['graph_data']=graph_data

    client_analysis_list.append(client_sentimental_analysis)
    
    specific_branch_data['client_analysis']=client_analysis_list
    
    



    #competitor_analysis
    comp_obj = competitor_reviews.objects.values('date__year', 'date__month')\
                                                     .annotate(
                                                                survey_date = F('date'),
                                                                value = twoDecimal(Avg('rating')),
                                                                                                                                  )
    comp_obj = pd.DataFrame(comp_obj)
    comp_obj['title'] = comp_obj['survey_date'].apply(lambda x : datetime.strptime(str(x)[:10],'%Y-%m-%d').strftime('%b-%Y'))
    comp_obj = comp_obj.to_dict(orient='records')
    competitor_analysis['heading']="Rating Trend"
    competitor_analysis["graph_type"]="bar"
    competitor_analysis['source']="Google"
    competitor_analysis['graph_data']=comp_obj
    competitor_analysis_list.append(competitor_analysis)
    # calculating sentimental percentage 
    count= comp_reviews.count()
    comp_positive=comp_reviews.filter(sentiment="Positive").count()
    comp_negative=comp_reviews.filter(Q(sentiment="Extreme") | Q(sentiment="Negative")).count()
    comp_positive_percentage=round(((comp_positive/count)*100),2)
    comp_negative_percentage=round(((comp_negative/count)*100),2)
    competitor_sentimental_analysis['heading']="sentimental analysis"
    competitor_sentimental_analysis["graph_type"]="bar"
    competitor_sentimental_analysis['source']="Google"
    graph_data=[{
                "title": "Positive",
                "value": comp_positive_percentage
              },
              {
                "title": "Negative",
                "value": comp_negative_percentage
              }]
    competitor_sentimental_analysis['graph_data']=graph_data
    competitor_analysis_list.append(competitor_sentimental_analysis)
    specific_branch_data['competitor_analysis']=competitor_analysis_list
    # fetching reviews
    review_data=review.values("id","date","review","rating","name","source","sentiment")
    review_data=pd.DataFrame(review_data)
    review_data['date'] = review_data['date'].apply(lambda x : datetime.strptime(str(x)[:10],'%Y-%m-%d').strftime('%d %b %Y'))
    review_data = review_data.to_dict(orient='records')

    
    specific_branch_data['reviews']=review_data
    
    specific_branch_data['swot_data']=swot_data
    specific_branch_data['actions_required']=actions_required
    specific_branch_data['topics']=topics

    res['pageName']= "Dashboard"
    res['all_branch_data']=[all_branch_data]
    


    return Response(res)


#fetching google page data

@api_view(['POST'])
def google_dashboard(request):
    data=request.data
    client_id=data['client_id']
        
        
    res={}
    rating_lis=[]
    all_branch_data_list=[]
    all_branch_data={}
    specific_branch_data={}
    avg_rating=0.0
    all_branch_data['branch_name']="Overall"

    gr_obj = reviews.objects.filter(client_id= client_id).values().order_by('-date')
    start_5 = gr_obj.filter(rating=5).count()
    start_4 = gr_obj.filter(rating=4).count()
    start_3 = gr_obj.filter(rating=3).count()
    start_2 = gr_obj.filter(rating=2).count()
    start_1 = gr_obj.filter(rating=1).count()
    star_values=[start_5,start_4,start_3,start_2,start_1]
    def calc_rating(x):
        
        rating_lis.append(int(x))
        return(int(x))
        
    total = gr_obj.count()
    gr_objj=pd.DataFrame(gr_obj)
    gr_objj['rating'] = gr_objj['rating'].apply(calc_rating)
    gr_objj = gr_objj.to_dict(orient='records')
    avg_rating = round(sum(rating_lis)/total,2)

    try:
        star_5_percentage = round(start_5*100/total,2)
    except:
        star_5_percentage = 0
    try:
        star_4_percentage = round(start_4*100/total,2)
    except:
        star_4_percentage = 0
    try:
        star_3_percentage = round(start_3*100/total,2)
    except:
        star_3_percentage = 0
    try:
        star_2_percentage = round(start_2*100/total,2)
    except:
        star_2_percentage = 0
    try:
        star_1_percentage = round(start_1*100/total,2)
    except:
        star_1_percentage = 0
    
    star_percentages=[star_5_percentage,star_4_percentage,star_3_percentage,star_2_percentage,star_1_percentage]

    star_rating = {
                        'total':total,
                        'net_rate':avg_rating,
                        'star':round(avg_rating),
                        "star_values":star_values,
                        "star_percentages":star_percentages,
                    }
    #calculating nss and nss_pie
    sentiment_positive = gr_obj.filter(sentiment='Positive').count()
    sentiment_neutral = gr_obj.filter(sentiment='Neutral').count()
    sentiment_negative = gr_obj.filter(sentiment='Negative').count()
    sentiment_extreme = gr_obj.filter(sentiment='Extreme').count()

    try:
        sentiment_positive_percentage  = round(sentiment_positive*100/total,2)
    except:
        sentiment_positive_percentage = 0
    try:
        sentiment_neutral_percentage  = round(sentiment_neutral*100/total,2)
    except:
        sentiment_neutral_percentage = 0
    try:
        sentiment_negative_percentage  = round(sentiment_negative*100/total,2)
    except:
        sentiment_negative_percentage = 0
    try:
        sentiment_extreme_percentage  = round(sentiment_extreme*100/total,2)
    except:
        sentiment_extreme_percentage = 0
    nss = sentiment_positive_percentage - sentiment_negative_percentage - sentiment_extreme_percentage
    nss = round(nss,2) if nss > 0 else 0
    sentiment_card = {
                    "nss": {
                                "nss_score": nss,
                                "total": total,
                                "positive": sentiment_positive_percentage,
                                "total_positive": sentiment_positive,
                                "negative": sentiment_negative_percentage,
                                "total_negative": sentiment_negative,
                                "extreme": sentiment_extreme_percentage,
                                "total_extreme": sentiment_extreme,
                                "neutral": sentiment_neutral_percentage,
                                "total_neutral": sentiment_neutral
                        },
                    "nss_pie": [
                                    {
                                        "label": "Positive",
                                        "percentage": sentiment_positive_percentage,
                                        "color": "#00AC69"
                                    },
                                    {
                                        "label": "Negative",
                                        "percentage": sentiment_negative_percentage,
                                        "color": "#EE6123"
                                    },
                                    {
                                        "label": "Extreme",
                                        "percentage": sentiment_extreme_percentage,
                                        "color": "#DB2B39"
                                    },
                                    {
                                        "label": "Neutral",
                                        "percentage": sentiment_neutral_percentage,
                                        "color": "#939799"
                                    }
                                ]
                        }
    #stats_cards
    surveyed = gr_obj.count()
    comments = gr_obj.exclude(review = '').count()
    alerts = gr_obj.filter(sentiment = 'Extreme').count()
    stats_res = [
                    {
                        'title':"Surveyed",
                        'value':surveyed
                    },
                    {
                        'title':"Comments",
                        'value':comments
                    },
                    {
                        'title':"Alerts",
                        'value':alerts
                    },
                    
    ]
    #calculating Rating over time
    rat_gr_obj = reviews.objects.values('date__year', 'date__month')\
                                                     .annotate(
                                                                count=Count('pk'),
                                                                year = F('date__year'),
                                                                survey_date = F('date'),
                                                                avg_rating = twoDecimal(Avg('rating')),
                                                                positive = twoDecimal((Cast(Sum(Case(
                                                                            When(sentiment='Positive',then=1),
                                                                            default=0,
                                                                            output_field=IntegerField()
                                                                            )),FloatField()))),
                                                                negative = twoDecimal((Cast(Sum(Case(
                                                                            When(sentiment='Negative',then=1),
                                                                            default=0,
                                                                            output_field=IntegerField()
                                                                            )),FloatField()))),
                                                                neutral = twoDecimal((Cast(Sum(Case(
                                                                            When(sentiment='Neutral',then=1),
                                                                            default=0,
                                                                            output_field=IntegerField()
                                                                            )),FloatField()))),
                                                                extreme = twoDecimal((Cast(Sum(Case(
                                                                            When(sentiment='Extreme',then=1),
                                                                            default=0,
                                                                            output_field=IntegerField()
                                                                            )),FloatField()))),
                                                                nss_abs = twoDecimal((F('positive')-F('negative')-F('extreme'))/Cast(Count('id'),FloatField())*100),
                                                                nss = Case(
                                                                            When(
                                                                                nss_abs__lt = 0,
                                                                                then = 0    
                                                                                ),
                                                                                default=F('nss_abs'),
                                                                                output_field=FloatField()
                                                                            )                                                                
                                                                  )
                                                     
                                                                                                                                                                          
    rat_gr_obj = pd.DataFrame(rat_gr_obj)
    rat_gr_obj['SURVEY_MONTH'] = rat_gr_obj['survey_date'].apply(lambda x : datetime.strptime(str(x)[:10],'%Y-%m-%d').strftime('%b-%Y'))
    rat_gr_obj['month']=rat_gr_obj['survey_date'].apply(lambda x : datetime.strptime(str(x)[:10],'%Y-%m-%d').strftime('%b-%y'))
    rat_gr_obj = rat_gr_obj.to_dict(orient='records')

    # fetching reviews
    review_data=gr_obj.values("id","date","review","rating","name","source","sentiment")
    review_data=pd.DataFrame(review_data)
    review_data['date'] = review_data['date'].apply(lambda x : datetime.strptime(str(x)[:10],'%Y-%m-%d').strftime('%d %b %Y'))
    review_data = review_data.to_dict(orient='records')
    
    #fetchin alerts
    alertss = gr_obj.filter(sentiment='Extreme').exclude(review = "").values('id','name','review','rating','date','sentiment','source').order_by('-date')
    if len(gr_obj)>0:
        alertss = pd.DataFrame(alertss)
        alertss['date'] = alertss['date'].apply(lambda x : datetime.strptime(str(x)[:10],'%Y-%m-%d').strftime('%b %Y'))
        alertss = alertss.to_dict(orient='records')
    else:
        alertss = []
    
    

          
          
    res['pageName']="Google Dashbaord"
    all_branch_data['branch_name']="overall"
    specific_branch_data['star_rating']=star_rating
    specific_branch_data['sentiment_card']=sentiment_card
    specific_branch_data['stats_cards']=stats_res
    specific_branch_data['rating_over_time']= rat_gr_obj
    specific_branch_data['reviews']= review_data
    specific_branch_data['alerts']= alertss
    all_branch_data["specific_branch_data"]=specific_branch_data
    all_branch_data_list.append(all_branch_data)
    res["all_branch_data"]=all_branch_data_list

    return Response(res)


# @api_view(['POST'])
# def rating_over_time(request):
    
#     res = {
#             'status':True,
#             'status_code':200,
#             'title':'OK',
#             'message':'Data for rating overtime',
#             'data':{
#                     'rating_over_time':gr_obj
#                     }
#           }      
#     return Response(res)

# ------------------------------- Data feeding ---------------------------------------

@api_view(['POST'])
def store_data(request):
    #google_reviews.objects.filter(user_id=5).delete()
    # data = request.data
    source="Google"
    client_id=1
    branch_id="ABSN01"
    #df = pd.read_csv('srinidhi_sent.csv')
    df = pd.read_csv('srinidhi_sent.csv', on_bad_lines='skip')
    df.fillna('',inplace=True)
    df['rating'] = df['rating']
    # df['rating'] = df['rating'].apply(lambda x : eval(x.split(' ')[1]))
    # return Response(df.shape[0])
    for i in range(len(df)):
        name = df['name'][i]
        review = df['review'][i]
        rating = df['rating'][i]
        day, month, year = df['date'][i].split('-')
        date = f"{year}-{month}-{day}"

        sentiment = df['sentiment'][i]

        gr_obj = reviews(
                                    client_id = client_id,
                                    name = name,
                                    review = review,
                                    rating = rating,
                                    date = date,
                                    sentiment = sentiment,
                                    branch_id=branch_id,
                                    source=source
                                )
        gr_obj.save()
        print(name)
        print(review)
        print(rating)
        print(date)
        print(sentiment)
        
    
    gr=reviews.objects.filter(client_id=1).values()
    res={
            'obj':gr,
            "message":"saved successfully"
        }
    return Response(res)    


    # name = models.CharField(max_length=200)
    # review = models.TextField()
    # rating = models.TextField()
    # date = models.DateTimeField()
    # sentiment = models.CharField(max_length=50)
    # branch_id=models.TextField(blank=True)
    # source=models.TextField(blank=True)
    # competitor_branch_id=models.TextField(blank=True)

@api_view(['POST'])
def competitor_store_data(request):
    #google_reviews.objects.filter(user_id=5).delete()
    # data = request.data
    source="Google"
    branch_id="ABKD01"
    competitor_branch_id="ABSN01"
    #df = pd.read_csv('srinidhi_sent.csv')
    df = pd.read_csv('kadamba_sen.csv', on_bad_lines='skip')
    df.fillna('',inplace=True)
    # df['rating'] = df['rating']
    # df['rating'] = df['rating'].apply(lambda x : eval(x.split(' ')[1]))
    # return Response(df.shape[0])
    for i in range(len(df)):
        name = df['name'][i]
        review = df['review'][i]
        # rating = df['rating'][i]
        day, month, year = df['date'][i].split('-')
        date = f"{year}-{month}-{day}"

        sentiment = df['sentiment'][i]

        gr_obj = competitor_reviews(
                                    
                                    name = name,
                                    review = review,
                                    
                                    date = date,
                                    sentiment = sentiment,
                                    branch_id=branch_id,
                                    source=source,
                                    competitor_branch_id=competitor_branch_id
                                )
        gr_obj.save()
        print(name)
        print(review)
        # print(rating)
        print(date)
        print(sentiment)
        
    
    gr=competitor_reviews.objects.filter(competitor_branch_id="ABSN01").values()
    res={
            'obj':gr,
            "message":"saved successfully"
        }
    return Response(res)    