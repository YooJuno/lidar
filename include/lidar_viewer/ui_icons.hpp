#pragma once
#include <imgui.h>
#include <algorithm>
#include <cmath>

namespace lidar_viewer
{
enum class UiIcon {Select, Box, Fit, Copy, Trash, Undo, Redo, Eye, Focus, Plus, Settings, Play, Pause, Previous, Next, Folder, Live};

inline void DrawUiIcon(ImDrawList * draw, UiIcon icon, ImVec2 origin, float size, ImU32 color)
{
  const auto p=[&](float x,float y){return ImVec2(origin.x+x*size,origin.y+y*size);};
  const float stroke=1.5f*size/18.f;
  const auto line=[&](float x,float y,float a,float b){draw->AddLine(p(x,y),p(a,b),color,stroke);};
  switch (icon) {
    case UiIcon::Select:
      line(.2f,.12f,.8f,.56f);line(.8f,.56f,.51f,.59f);line(.51f,.59f,.38f,.86f);line(.38f,.86f,.2f,.12f);break;
    case UiIcon::Box:
      draw->AddRect(p(.18f,.25f),p(.82f,.82f),color,2,0,stroke);
      line(.18f,.25f,.36f,.1f);line(.36f,.1f,.95f,.1f);line(.95f,.1f,.95f,.65f);line(.95f,.65f,.82f,.82f);break;
    case UiIcon::Fit: case UiIcon::Focus:
      for (int x=0;x<2;++x) for (int y=0;y<2;++y) {
        const float a=x ? .84f : .16f,b=y ? .84f : .16f;
        line(a,b,a+(x ? -.23f : .23f),b);line(a,b,a,b+(y ? -.23f : .23f));
      }break;
    case UiIcon::Copy:
      draw->AddRect(p(.12f,.12f),p(.65f,.65f),color,2,0,stroke);
      draw->AddRectFilled(p(.33f,.33f),p(.88f,.88f),ImGui::GetColorU32(ImGuiCol_Button),2);
      draw->AddRect(p(.33f,.33f),p(.88f,.88f),color,2,0,stroke);break;
    case UiIcon::Trash:
      line(.16f,.28f,.84f,.28f);line(.35f,.14f,.65f,.14f);line(.26f,.36f,.3f,.87f);
      line(.3f,.87f,.7f,.87f);line(.7f,.87f,.74f,.36f);line(.43f,.42f,.43f,.73f);line(.57f,.42f,.57f,.73f);break;
    case UiIcon::Undo: case UiIcon::Redo: {
      const bool redo=icon==UiIcon::Redo;
      const auto q=[&](float x,float y){return p(redo ? 1-x : x,y);};
      draw->AddBezierCubic(q(.2f,.38f),q(.9f,-.05f),q(1.f,1.f),q(.45f,.82f),color,stroke);
      draw->AddLine(q(.2f,.38f),q(.22f,.13f),color,stroke);draw->AddLine(q(.2f,.38f),q(.46f,.42f),color,stroke);break;
    }
    case UiIcon::Eye:
      draw->AddBezierCubic(p(.08f,.5f),p(.3f,.08f),p(.7f,.08f),p(.92f,.5f),color,stroke);
      draw->AddBezierCubic(p(.08f,.5f),p(.3f,.92f),p(.7f,.92f),p(.92f,.5f),color,stroke);
      draw->AddCircle(p(.5f,.5f),size*.14f,color,16,stroke);break;
    case UiIcon::Plus:line(.5f,.15f,.5f,.85f);line(.15f,.5f,.85f,.5f);break;
    case UiIcon::Play:draw->AddTriangleFilled(p(.25f,.13f),p(.84f,.5f),p(.25f,.87f),color);break;
    case UiIcon::Pause:
      draw->AddRectFilled(p(.2f,.16f),p(.4f,.84f),color,1);draw->AddRectFilled(p(.6f,.16f),p(.8f,.84f),color,1);break;
    case UiIcon::Previous:line(.68f,.18f,.32f,.5f);line(.32f,.5f,.68f,.82f);break;
    case UiIcon::Next:line(.32f,.18f,.68f,.5f);line(.68f,.5f,.32f,.82f);break;
    case UiIcon::Folder:
      line(.1f,.28f,.1f,.82f);line(.1f,.82f,.9f,.82f);line(.9f,.82f,.9f,.34f);
      line(.9f,.34f,.48f,.34f);line(.48f,.34f,.38f,.19f);line(.38f,.19f,.1f,.19f);line(.1f,.19f,.1f,.28f);break;
    case UiIcon::Live:
      draw->AddCircleFilled(p(.5f,.5f),size*.12f,color,12);
      for (const float r:{.28f,.43f}) {
        draw->PathArcTo(p(.5f,.5f),size*r,-1.1f,1.1f,10);draw->PathStroke(color,0,stroke);
        draw->PathArcTo(p(.5f,.5f),size*r,2.04f,4.24f,10);draw->PathStroke(color,0,stroke);
      }break;
    case UiIcon::Settings:
      draw->AddCircle(p(.5f,.5f),size*.28f,color,16,stroke);
      draw->AddCircle(p(.5f,.5f),size*.10f,color,12,stroke);
      for (int i=0;i<8;++i) {
        const float a=i*3.14159265f*.25f,x=std::cos(a),y=std::sin(a);
        line(.5f+x*.28f,.5f+y*.28f,.5f+x*.42f,.5f+y*.42f);
      }break;
  }
}

inline bool ActionButton(const char * id,UiIcon icon,const char * label,ImVec2 size,float scale)
{
  if (size.y==0) size.y=ImGui::GetFrameHeight();
  const bool clicked=ImGui::Button(id,size);const auto a=ImGui::GetItemRectMin(),b=ImGui::GetItemRectMax();
  const float text_width=label ? ImGui::CalcTextSize(label).x : 0,icon_size=16*scale,gap=label ? 6*scale : 0;
  const float x=a.x+std::max(0.f,(b.x-a.x-icon_size-gap-text_width)*.5f);
  auto * draw=ImGui::GetWindowDrawList();const auto color=ImGui::GetColorU32(ImGuiCol_Text);
  DrawUiIcon(draw,icon,{x,a.y+(size.y-icon_size)*.5f},icon_size,color);
  if (label) draw->AddText({x+icon_size+gap,a.y+(size.y-ImGui::GetTextLineHeight())*.5f},color,label);
  return clicked;
}

inline bool IconButton(const char * id, UiIcon icon, const char * tip, float scale, bool selected=false,
  const char * label=nullptr)
{
  const float icon_size=18*scale, height=30*scale;
  const float label_width=label ? ImGui::CalcTextSize(label).x+8*scale : 0;
  if (selected) ImGui::PushStyleColor(ImGuiCol_Button,{.13f,.34f,.42f,1});
  const bool clicked=ImGui::Button(id,{height+label_width,height});
  if (selected) ImGui::PopStyleColor();
  const auto origin=ImGui::GetItemRectMin();
  const auto color=ImGui::GetColorU32(ImGuiCol_Text);
  DrawUiIcon(ImGui::GetWindowDrawList(),icon,{origin.x+6*scale,origin.y+6*scale},icon_size,color);
  if (label) ImGui::GetWindowDrawList()->AddText({origin.x+30*scale,origin.y+(height-ImGui::GetTextLineHeight())*.5f},color,label);
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
    ImGui::BeginTooltip();ImGui::PushTextWrapPos(ImGui::GetFontSize()*26);
    ImGui::TextUnformatted(tip);ImGui::PopTextWrapPos();ImGui::EndTooltip();
  }
  return clicked;
}
}  // namespace lidar_viewer
