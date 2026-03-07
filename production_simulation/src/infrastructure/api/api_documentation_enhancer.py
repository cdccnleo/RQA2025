"""
API文档增强器

提供API文档增强功能
"""

class APIDocumentationEnhancer:
    """API文档增强器"""

    def __init__(self):
        self.enhancements = {}
        self.templates = {}

    def enhance_documentation(self, docs):
        """增强API文档"""
        enhanced = docs.copy()

        # 应用所有增强
        for name, enhancement in self.enhancements.items():
            try:
                enhanced = enhancement(enhanced)
            except Exception as e:
                print(f"应用增强 {name} 时出错: {e}")

        return enhanced

    def add_enhancement(self, name, enhancement_func):
        """添加增强功能"""
        self.enhancements[name] = enhancement_func

    def remove_enhancement(self, name):
        """移除增强功能"""
        if name in self.enhancements:
            del self.enhancements[name]
            return True
        return False

    def add_template(self, name, template):
        """添加模板"""
        self.templates[name] = template

    def apply_template(self, docs, template_name):
        """应用模板"""
        template = self.templates.get(template_name)
        if template:
            return template(docs)
        return docs

    def get_enhancement_count(self):
        """获取增强数量"""
        return len(self.enhancements)

    def get_template_count(self):
        """获取模板数量"""
        return len(self.templates)
