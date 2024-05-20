from pydantic import BaseModel, Field
from typing import List, Optional, Generic, TypeVar
from pydantic.generics import GenericModel

class AnnouncementRequest(BaseModel):
    market: Optional[str] = '' #深沪京 
    tabName: Optional[str] = None #公告
    plate: Optional[List[str]] = [] #板块
    category: Optional[List[str]] = None #公告分类
    industry: Optional[List[str]] = [] #行业
    stock: Optional[List[str]] = None #股票代码
    searchkey: Optional[str] = '' #标题关键字
    seDate: Optional[str] = None #起始时间

class Announcement(BaseModel):
    id: Optional[str] = None
    secCode: Optional[str] = None
    secName: Optional[str] = None
    orgId: Optional[str] = None
    announcementId: Optional[str] = None
    announcementTitle: Optional[str] = None
    announcementTime: Optional[int] = None
    adjunctUrl: Optional[str] = None
    adjunctSize: Optional[int] = None
    adjunctType: Optional[str] = None
    storageTime: Optional[str] = None
    columnId: Optional[str] = None
    pageColumn: Optional[str] = None
    announcementType: Optional[str] = None
    associateAnnouncement: Optional[str] = None
    important: Optional[str] = None
    batchNum: Optional[str] = None
    announcementContent: Optional[str] = None
    orgName: Optional[str] = None
    tileSecName: Optional[str] = None
    shortTitle: Optional[str] = None
    announcementTypeName: Optional[str] = None
    secNameList: Optional[str] = None

class AnnouncementResponse(BaseModel):
    data: List[Announcement] = []