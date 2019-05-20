from django.urls import path
from django.conf.urls import url
from . import views

app_name = 'articles'

urlpatterns = [

    url(r'^$', views.article_list, name='list'),
    # url(r'^about/$', views.article_list),
    url(r'^create/$', views.article_create, name='create'),
    url(r'^(?P<slug>[\w-]+)/$', views.article_detail, name="detail")

]


"""
    
    PERSONAL COMMENTS, DO NOT DELETE !!!!
    
    between ^ and $ is the regex
    to create a named capturing group syntax -> (?P< HERE_IS_THE_NAME_OF_THING_WE_WANT_TO_CAPTURE >
    
    now, what can this slug be, in the url -> it can be :
                    [\w] -> any kind of letter, number or underscore
                    [\w-] -> means all of the above plus the hyphen
                    [\w-]+ -> this thing can be of any length
                    
                    all of above will be capture in slug variable and it will be sent to the function article_detail
                      

"""
