# Bu dosyayı main.py'ye sonra kopyalayacağız
# Sadece import kısmına eklenecek:

# Admin routes ekle (main.py satır 270 civarı - app oluştuktan sonra)
from routes.admin_routes import router as admin_router
app.include_router(admin_router)
